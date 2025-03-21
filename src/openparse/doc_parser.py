from pathlib import Path
from typing import List, Literal, TypedDict, TypeVar, Union, Optional, Dict

from openparse import consts, tables, text
from openparse._types import NOT_GIVEN, NotGiven
from openparse.pdf import Pdf
from openparse.processing import (
    BasicIngestionPipeline,
    IngestionPipeline,
    NoOpIngestionPipeline,
)
from openparse.schemas import Node, ParsedDocument, TableElement, TextElement

from openparse.schemas import ImageElement
from openparse.processing.markitdown_doc_parser import DocumentParser as MarkItDownParser
from openparse.config import config, Config

import zipfile
import tempfile
import shutil
import boto3
from urllib.parse import urlparse
import os
import logging

logger = logging.getLogger(__name__)

IngestionPipelineType = TypeVar("IngestionPipelineType", bound=IngestionPipeline)


class UnitableArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["unitable"]
    min_table_confidence: float
    table_output_format: Literal["html"]


class TableTransformersArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["table-transformers"]
    min_table_confidence: float
    min_cell_confidence: float
    table_output_format: Literal["markdown", "html"]


class PyMuPDFArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["pymupdf"]
    table_output_format: Literal["markdown", "html"]


def _table_args_dict_to_model(
    args_dict: Union[TableTransformersArgsDict, PyMuPDFArgsDict],
) -> Union[tables.TableTransformersArgs, tables.PyMuPDFArgs]:
    if args_dict["parsing_algorithm"] == "table-transformers":
        return tables.TableTransformersArgs(**args_dict)
    elif args_dict["parsing_algorithm"] == "pymupdf":
        return tables.PyMuPDFArgs(**args_dict)
    elif args_dict["parsing_algorithm"] == "unitable":
        return tables.UnitableArgs(**args_dict)
    else:
        raise ValueError(
            f"Unsupported parsing_algorithm: {args_dict['parsing_algorithm']}"
        )


class DocumentParser:
    """
    A parser for extracting elements from PDF documents, including text and tables.

    Attributes:
        processing_pipeline (Optional[IngestionPipelineType]): A subclass of IngestionPipeline to process extracted elements.
        table_args (Optional[Union[TableTransformersArgsDict, PyMuPDFArgsDict]]): Arguments to customize table parsing.
    """

    _verbose: bool = False

    def __init__(
        self,
        *,
        processing_pipeline: Union[IngestionPipeline, NotGiven, None] = NOT_GIVEN,
        # table_args: Union[TableTransformersArgsDict, PyMuPDFArgsDict, NotGiven] = NOT_GIVEN,
        table_args=None,
        use_markitdown: bool = True,
        llm_client: Optional[object] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 50,
        use_tokens: bool = True,
        verbose: bool = False,
        **kwargs
    ):
        self._verbose = verbose
        
        # Initialize processing pipeline
        self.processing_pipeline: IngestionPipeline
        if processing_pipeline is NOT_GIVEN:
            self.processing_pipeline = BasicIngestionPipeline()
        elif processing_pipeline is None:
            self.processing_pipeline = NoOpIngestionPipeline()
        else:
            self.processing_pipeline = processing_pipeline

        # Set pipeline verbosity
        self.processing_pipeline.verbose = self._verbose
        
        """
        Initialize the document parser.
        
        Config Args:
            use_markitdown: Whether to use MarkItDown for document parsing
            use_ocr: Whether to use OCR for document parsing
            table_args: Arguments for table extraction
            processing_pipeline: Pipeline for processing extracted nodes
            llm_client: Optional LLM client for enhanced parsing
            chunk_size: Maximum size of text chunks (in characters or tokens)
            chunk_overlap: Number of characters or tokens to overlap between chunks
            use_tokens: If True, measures length in tokens; if False, uses characters
            verbose: Whether to enable verbose logging
        """

        # Extract config values from kwargs if present
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            logger.info(f"Received config: {kwargs['config']}")
            logger.debug(f"Config chunking settings: {kwargs['config'].get('chunking', {})}")
            config_dict = kwargs['config']
            # Extract chunking parameters
            chunking = config_dict.get('chunking', {})
            min_tokens = chunking.get('minTokens', config_dict.get('minTokens', 125))
            max_tokens = chunking.get('maxTokens', config_dict.get('maxTokens', 256))
            chunk_overlap = chunking.get('overlap', config_dict.get('chunkOverlap', 50))
            use_tokens = chunking.get('useTokens', config_dict.get('useTokens', True))
            
            # Override parameters with values from config
            chunk_size = max_tokens
            chunk_overlap = chunk_overlap
            use_tokens = use_tokens
            use_markitdown = config_dict.get('use_markitdown', use_markitdown)
            
            logger.info(f"Using config parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, use_tokens={use_tokens}")

        self.use_markitdown = use_markitdown
        self.table_args = table_args
        
        if use_markitdown:
            self.markitdown_parser = MarkItDownParser(
                llm_client=llm_client,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_tokens=use_tokens
            )
            logger.info(f"Initialized MarkItDown parser with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, use_tokens={use_tokens}")
    def _process_directory(
        self,
        files: List[Path],
        batch_size: int
    ) -> List[ParsedDocument]:
        """Process directory of files in batches."""
        results = self.markitdown_parser.parse_batch(files, batch_size)
        return [
            ParsedDocument(
                nodes=nodes,
                filename=file_path.name,  # Use file_path from enumerate
                num_pages=1,
                coordinate_system=consts.COORDINATE_SYSTEM,
                table_parsing_kwargs=None,
                **metadata
            )
            for file_path, (nodes, metadata) in zip(files, results)
        ]

    def _process_markitdown(
        self,
        file_path: Path,
        nodes: List[Node],
        metadata: Dict
    ) -> ParsedDocument:
        """Process single file with MarkItDown."""
        # Process nodes through pipeline if configured
        if self.processing_pipeline:
            nodes = self.processing_pipeline.run(nodes)

        # Use page_count directly from metadata since it's already set
        num_pages = metadata.get('page_count', 1)

        return ParsedDocument(
            nodes=nodes,  # Make sure we're passing the processed nodes
            filename=file_path.name,
            num_pages=num_pages,
            coordinate_system=consts.COORDINATE_SYSTEM,
            table_parsing_kwargs=None,
            creation_date=metadata.get('creation_date'),
            last_modified_date=metadata.get('last_modified_date'),
            last_accessed_date=metadata.get('last_accessed_date'),
            file_size=metadata.get('file_size')
        )

    def _process_pdfminer(
        self,
        file: Union[str, Path],
        parse_elements: Optional[Dict[str, bool]],
        embeddings_provider: Optional[str],
        ocr: bool
    ) -> ParsedDocument:
        """Process file with PDFMiner."""
        temp_config = self._update_config(parse_elements, embeddings_provider)
        doc = Pdf(file)
        nodes = self._extract_nodes(doc, ocr, temp_config)
        return ParsedDocument(
            nodes=nodes,
            filename=Path(file).name,
            num_pages=doc.num_pages,
            coordinate_system=consts.COORDINATE_SYSTEM,
            table_parsing_kwargs=self._get_table_kwargs(),
            **doc.file_metadata
        )

    def _update_config(
        self,
        parse_elements: Optional[Dict[str, bool]],
        embeddings_provider: Optional[str]
    ) -> Config:
        """Update config with overrides."""
        temp_config = config
        if parse_elements:
            temp_config._parse_elements.update(parse_elements)
        if embeddings_provider:
            temp_config._embeddings_provider = embeddings_provider
        return temp_config


    def _extract_nodes(
        self,
        doc: Pdf,
        ocr: bool,
        temp_config: Config
    ) -> List[Node]:
        """Extract and process nodes from document."""
        text_nodes = self._extract_text_nodes(doc, ocr)
        table_nodes = self._extract_table_nodes(doc, temp_config)
        nodes = text_nodes + table_nodes
        return self.processing_pipeline.run(nodes)

    def _extract_text_nodes(self, doc: Pdf, ocr: bool) -> List[Node]:
        """Extract text nodes from document."""
        text_engine: Literal["pdfminer", "pymupdf"] = (
            "pdfminer" if not ocr else "pymupdf"
        )
        text_elems = text.ingest(doc, parsing_method=text_engine)
        return self._elems_to_nodes(text_elems)


    def _extract_table_nodes(
        self,
        doc: Pdf,
        temp_config: Config
    ) -> List[Node]:
        """Extract table nodes if enabled."""
        if not self.table_args or not temp_config._parse_elements.get("tables", True):
            return []
        table_args_obj = _table_args_dict_to_model(self.table_args)
        table_elems = tables.ingest(doc, table_args_obj, verbose=self._verbose)
        return self._elems_to_nodes(table_elems)

    def _get_table_kwargs(self) -> Optional[Dict]:
        """Get table kwargs if table args present."""
        if not hasattr(self, 'table_args_obj'):
            return None
        return self.table_args_obj.model_dump()

    def _get_s3_client(self, endpoint_url: Optional[str] = None) -> boto3.client:
        """Create and return an S3 client configured for either AWS S3 or Cloudflare R2."""
        # Default to R2 configuration if endpoint_url is not provided
        if not endpoint_url:
            account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
            if not account_id:
                raise ValueError("CLOUDFLARE_ACCOUNT_ID environment variable is required for R2 storage")
            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

        # Log the configuration being used
        logger.info("🟠☁️ Creating S3 client with configuration:")
        logger.info(f"Endpoint URL: {endpoint_url}")

        client_kwargs = {
            'service_name': 's3',
            'endpoint_url': endpoint_url,
            'aws_access_key_id': os.getenv('R2_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('R2_SECRET_ACCESS_KEY'),
            'region_name': os.getenv('R2_REGION', 'auto'),
            'config': boto3.Config(
                request_checksum_calculation='WHEN_REQUIRED',
                response_checksum_validation='WHEN_REQUIRED'
            )
        }

        # Log the actual values being used (without exposing secrets)
        logger.info("S3 client kwargs:")
        safe_kwargs = client_kwargs.copy()
        if 'aws_secret_access_key' in safe_kwargs:
            safe_kwargs['aws_secret_access_key'] = '***'
        logger.info(f"Client kwargs: {safe_kwargs}")

        # Remove None values
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
        
        return boto3.client(**client_kwargs)

    def _download_from_s3(self, s3_url: str) -> Path:
        """Download file from S3/R2 to local temporary file."""
        parsed_url = urlparse(s3_url)
        bucket = parsed_url.netloc
        key = parsed_url.path.lstrip('/')
        logger.info(f"Parsed S3 details - Bucket: {bucket}, Key: {key}")
        
        # Extract file extension
        file_ext = ''
        if '.' in key:
            file_ext = '.' + key.split('.')[-1].lower()
        
        logger.info(f"File extension: {file_ext}")
        if not file_ext:
            logger.warning(f"No file extension found for key: {key}, attempting to infer from content")
            # Default to PDF if we can't determine the extension
            file_ext = '.pdf'
        
        if file_ext not in {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.txt', '.md', '.rst', '.zip'}:
            logger.error(f"Unsupported file extension: {file_ext}")
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        # Create S3/R2 client
        try:
            s3_client = self._get_s3_client()
            logger.info("Successfully created S3/R2 client")
        except Exception as e:
            logger.error(f"Failed to create S3/R2 client: {str(e)}")
            raise
        
        # Create temporary file with proper extension
        temp_file = Path(tempfile.mktemp(suffix=file_ext))
        
        try:
            s3_client.download_file(bucket, key, str(temp_file))
            logger.info(f"Downloaded S3 file to {temp_file}")
            return temp_file
        except Exception as e:
            logger.error(f"Failed to download S3 file: {str(e)}")
            if temp_file.exists():
                temp_file.unlink()
            raise ValueError(f"Failed to download S3 file: {str(e)}")

    def parse(
        self,
        file: Union[str, Path],
        ocr: bool = False,
        parse_elements: Optional[Dict[str, bool]] = None,
        embeddings_provider: Optional[Literal["openai", "ollama", "cloudflare"]] = None,
        batch_size: int = 1
    ) -> Union[ParsedDocument, List[ParsedDocument]]:
        """Parse document using configured parser."""
        logger.info(f"Starting parse for file: {file}")
        
        # Handle S3/R2 URLs
        if isinstance(file, str) and file.startswith('s3://'):
            logger.info(f"Detected S3/R2 URL: {file}")
            try:
                temp_file = self._download_from_s3(file)
                logger.info(f"Successfully downloaded S3/R2 file to: {temp_file}")
                file_path = temp_file
            except Exception as e:
                logger.error(f"Failed to process S3/R2 file: {str(e)}")
                raise ValueError(f"Failed to process S3/R2 file: {str(e)}")
        else:
            logger.info(f"Processing local file: {file}")
            file_path = Path(file)
        
        try:
            if self.use_markitdown:
                if file_path.is_dir():
                    files = list(file_path.glob("*"))
                    return self._process_directory(files, batch_size)
                elif file_path.suffix.lower() == '.zip':
                    # Extract files from ZIP and process each separately
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        files = []
                        for filename in zip_ref.namelist():
                            # Extract to temporary directory
                            temp_dir = Path(tempfile.mkdtemp())
                            extracted_path = temp_dir / Path(filename).name
                            with zip_ref.open(filename) as source, open(extracted_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            files.append(extracted_path)
                        
                        # Process extracted files
                        try:
                            return self._process_directory(files, batch_size)
                        finally:
                            # Clean up temp files
                            shutil.rmtree(temp_dir)
                
                nodes, metadata = self.markitdown_parser.parse(file_path)
                return self._process_markitdown(file_path, nodes, metadata)
                
            return self._process_pdfminer(
                file_path,
                parse_elements,
                embeddings_provider,
                ocr
            )
        finally:
            # Clean up temporary file if it was downloaded from S3
            if isinstance(file, str) and file.startswith('s3://'):
                temp_file.unlink()

    @staticmethod
    def _elems_to_nodes(
        elems: Union[List[TextElement], List[TableElement], List[ImageElement]],
    ) -> List[Node]:
        return [
            Node(
                elements=(e,),
            )
            for e in elems
        ]
