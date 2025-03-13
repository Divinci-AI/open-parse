"""
Tests for DocumentParser class in OpenParse.

This test suite verifies:
- Parser initialization
- PDF document processing
- MarkItDown integration
- Configuration handling
- Error cases

Run tests with:
    pytest test_docparser_0_7_2.py -v
    
With coverage:
    pytest test_docparser_0_7_2.py -v --cov=openparse

Requirements:
    pip install pytest pytest-cov
    pip install openparse[test]
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import date
import zipfile
import io

from openparse import DocumentParser, ParsedDocument, Node
from openparse.processing import BasicIngestionPipeline, NoOpIngestionPipeline
from openparse.schemas import TextElement, Bbox, NodeVariant

@pytest.fixture
def sample_pdf():
    return Path(__file__).parent / "sample_data" / "pdf-with-image.pdf"

@pytest.fixture
def sample_docx():
    return Path(__file__).parent / "sample_data" / "docx-test-data.docx"

@pytest.fixture
def mock_markitdown_result():
    """Create a standard mock result from MarkItDown."""
    return MagicMock(text_content="Test content")

@pytest.fixture
def sample_zip(tmp_path):
    """Create a sample ZIP file with test documents."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add a PDF file
        zf.writestr("test.pdf", TestDocumentParser.create_test_pdf())
        # Add a DOCX file
        zf.writestr("test.docx", TestDocumentParser.create_test_docx())
    return zip_path

@pytest.fixture
def mock_markitdown():
    """Create a mock MarkItDown instance."""
    with patch('openparse.processing.markitdown_doc_parser.MarkItDown') as mock:
        mock_instance = mock.return_value
        mock_instance.convert_local.return_value.text_content = "Test content"
        yield mock

class TestMarkItDownDocParser:
    """Tests specific to MarkItDown document parser functionality."""
    
    def test_supported_formats(self):
        """Test supported file format detection."""
        parser = DocumentParser(use_markitdown=True)
        supported = {'.pdf', '.docx', '.pptx', '.xlsx', '.html', '.txt', '.json', '.xml', '.zip'}
        assert parser.markitdown_parser.SUPPORTED_FORMATS == supported

    def test_text_to_nodes_conversion(self):
        """Test conversion of text content to nodes."""
        parser = DocumentParser(use_markitdown=True)
        text = "Test content\nMultiple lines\nMore content"
        nodes = parser.markitdown_parser._text_to_nodes(text)
        
        assert len(nodes) > 0
        assert isinstance(nodes[0], Node)
        assert isinstance(nodes[0].elements[0], TextElement)
        assert "Test content" in nodes[0].elements[0].text

    def test_metadata_extraction(self, tmp_path):
        """Test file metadata extraction."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        parser = DocumentParser(use_markitdown=True)
        metadata = parser.markitdown_parser._get_metadata(None, test_file)
        
        assert isinstance(metadata, dict)
        assert "creation_date" in metadata
        assert "last_modified_date" in metadata
        assert "last_accessed_date" in metadata
        assert "file_size" in metadata
        assert "file_type" in metadata
        assert metadata["file_type"] == ".txt"

    def test_split_text_with_overlap(self):
        """Test text chunking with overlap."""
        parser = DocumentParser(
            use_markitdown=True, 
            chunk_size=50,
            chunk_overlap=10,
            use_tokens=False
        )
        
        # Test with a long text that needs chunking
        long_text = "This is a test " * 20  # Creates text longer than chunk_size
        chunks = parser.markitdown_parser.split_text_with_overlap(long_text)
        
        # Verify we got multiple chunks
        assert len(chunks) > 1
        
        # Verify chunk size constraints
        for chunk in chunks:
            assert len(chunk) <= 50 + 10  # Allow for some overlap
            
        # Verify overlap exists between chunks
        if len(chunks) >= 2:
            # The end of first chunk should appear at the start of second chunk
            overlap = chunks[0][-10:]
            assert overlap in chunks[1]

    def test_token_based_chunking(self):
        """Test token-based text chunking."""
        parser = DocumentParser(
            use_markitdown=True, 
            chunk_size=10,  # 10 tokens
            chunk_overlap=2,
            use_tokens=True
        )
        
        # Create text with exactly 25 tokens
        text = "one two three four five six seven eight nine ten " * 2 + "one two three four five"
        chunks = parser.markitdown_parser.split_text_with_overlap(text)
        
        # Should create 3 chunks with our settings
        assert len(chunks) >= 2
        
        # Check token counts in each chunk
        for chunk in chunks:
            token_count = len([word for word in chunk.split() if word])
            assert token_count <= 12  # chunk_size + overlap

    def test_batch_processing(self, tmp_path):
        """Test batch processing of multiple files."""
        # Create test files
        files = []
        for i in range(3):
            file = tmp_path / f"test{i}.txt"
            file.write_text(f"content {i}")
            files.append(file)
        
        parser = DocumentParser(use_markitdown=True)
        results = parser.markitdown_parser.parse_batch(files, batch_size=2)
        
        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (nodes, metadata)

    def test_zip_processing(self, sample_zip):
        """Test processing of ZIP files."""
        parser = DocumentParser(use_markitdown=True)
        nodes, metadata = parser.markitdown_parser.parse(sample_zip)
        
        assert isinstance(nodes, list)
        assert isinstance(metadata, dict)
        assert metadata["is_zip"] is True
        assert metadata["file_type"] == ".zip"

    def test_unsupported_format(self, tmp_path):
        """Test error handling for unsupported file formats."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("test content")
        
        parser = DocumentParser(use_markitdown=True)
        with pytest.raises(ValueError, match="Unsupported file format"):
            parser.markitdown_parser.parse(unsupported_file)

    def test_error_handling_in_batch(self, tmp_path):
        """Test error handling during batch processing."""
        # Create one valid and one invalid file
        valid_file = tmp_path / "valid.txt"
        invalid_file = tmp_path / "invalid.xyz"
        valid_file.write_text("valid content")
        invalid_file.write_text("invalid content")
        
        parser = DocumentParser(use_markitdown=True)
        results = parser.markitdown_parser.parse_batch([valid_file, invalid_file])
        
        # Should only get result from valid file
        assert len(results) == 1
        assert isinstance(results[0], tuple)

    @pytest.mark.parametrize("file_type", [".pdf", ".docx", ".txt"])
    def test_different_file_types(self, tmp_path, file_type):
        """Test processing of different supported file types."""
        test_file = tmp_path / f"test{file_type}"
        
        if file_type == ".pdf":
            test_file.write_bytes(TestDocumentParser.create_test_pdf())
        elif file_type == ".docx":
            test_file.write_bytes(TestDocumentParser.create_test_docx())
        else:
            test_file.write_text("test content")
        
        parser = DocumentParser(use_markitdown=True)
        nodes, metadata = parser.markitdown_parser.parse(test_file)
        
        assert isinstance(nodes, list)
        assert isinstance(metadata, dict)
        assert metadata["file_type"] == file_type

    def test_llm_client_initialization(self):
        """Test initialization with LLM client."""
        mock_llm = MagicMock()
        parser = DocumentParser(use_markitdown=True, llm_client=mock_llm)
        
        # Verify the LLM client was passed to MarkItDown
        assert parser.markitdown_parser.parser is not None
        
    def test_custom_chunking_parameters(self):
        """Test initialization with custom chunking parameters."""
        parser = DocumentParser(
            use_markitdown=True,
            chunk_size=2000,
            chunk_overlap=300,
            use_tokens=False
        )
        
        assert parser.markitdown_parser.chunk_size == 2000
        assert parser.markitdown_parser.chunk_overlap == 300
        assert parser.markitdown_parser.use_tokens is False

class TestDocumentParser:
    def test_init_default(self):
        """Test default initialization."""
        parser = DocumentParser()
        assert parser._verbose is False
        assert isinstance(parser.processing_pipeline, BasicIngestionPipeline)
        assert parser.table_args is None
        assert parser.use_markitdown is False

    def test_init_with_markitdown(self, mock_markitdown):
        """Test initialization with MarkItDown."""
        parser = DocumentParser(use_markitdown=True)
        assert parser.use_markitdown is True
        assert hasattr(parser, 'markitdown_parser')

    @pytest.mark.parametrize("verbose", [True, False])
    def test_verbose_setting(self, verbose):
        """Test verbose flag propagation."""
        parser = DocumentParser(verbose=verbose)
        assert parser._verbose == verbose
        assert parser.processing_pipeline.verbose == verbose

    def test_parse_pdf(self, sample_pdf):
        """Test PDF parsing."""
        parser = DocumentParser()
        result = parser.parse(sample_pdf)
        assert isinstance(result, ParsedDocument)
        assert result.filename == sample_pdf.name
        assert len(result.nodes) > 0

    def test_parse_with_markitdown(self, sample_docx, mock_markitdown):
        """Test non-PDF parsing with MarkItDown."""
        mock_nodes = [Node(elements=(TextElement(
            text="Test content",
            lines=(),
            bbox=Bbox(
                page=1,
                page_height=1000,
                page_width=1000,
                x0=0, y0=0,
                x1=1000, y1=1000
            ),
            variant=NodeVariant.TEXT
        ),))]

        mock_metadata = {
            "creation_date": None,
            "last_modified_date": date.today(),
            "last_accessed_date": date.today(),
            "file_size": 100,
            "file_type": ".docx",
            "page_count": 1
        }

        # Create a parser with mocked markitdown_parser and NoOp pipeline
        parser = DocumentParser(
            use_markitdown=True,
            processing_pipeline=NoOpIngestionPipeline()
        )
        
        # Mock the parse method of markitdown_parser to return our mock data
        parser.markitdown_parser.parse = MagicMock(return_value=(mock_nodes, mock_metadata))
        
        result = parser.parse(sample_docx)

        assert isinstance(result, ParsedDocument)
        assert result.filename == sample_docx.name
        assert len(result.nodes) == 1
        assert result.nodes[0].elements[0].text == "Test content"

    @staticmethod
    def create_test_docx():
        """Create valid test DOCX."""
        content = (
            b"PK\x03\x04\x14\x00\x06\x00\x08\x00\x00\x00!\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13\x00\x00\x00"
            b"[Content_Types].xml"
        )
        return content

    @staticmethod
    def create_test_pdf():
        """Create minimal valid PDF with correct /Pages tree."""
        return (
            b"%PDF-1.7\n"
            b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n"
            b"2 0 obj\n<</Type/Pages/Count 1/Kids[3 0 R]>>\nendobj\n"
            b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>\nendobj\n"
            b"xref\n"
            b"0 4\n"
            b"0000000000 65535 f\n"
            b"0000000009 00000 n\n"
            b"0000000056 00000 n\n" 
            b"0000000111 00000 n\n"
            b"trailer\n"
            b"<</Size 4/Root 1 0 R>>\n"
            b"startxref\n"
            b"180\n"
            b"%%EOF"
        )

    def test_directory_processing(self, tmp_path, mock_markitdown):
        """Test directory batch processing."""
        # Create test files
        doc1 = tmp_path / "test1.docx" 
        doc2 = tmp_path / "test2.pdf"
        
        doc1.write_bytes(self.create_test_docx())
        doc2.write_bytes(self.create_test_pdf())

        # Configure mock
        mock_markitdown.return_value.parse.return_value = (
            [Node(elements=(TextElement(
                text="Test content",
                lines=(),
                bbox=Bbox(
                    page=1,
                    page_height=1000,
                    page_width=1000,
                    x0=0, y0=0,
                    x1=1000, y1=1000
                ),
                variant=NodeVariant.TEXT
            ),))],
            {
                "creation_date": None,
                "last_modified_date": date.today(),
                "last_accessed_date": date.today(),
                "file_size": 100,
                "file_type": ".docx"
            }
        )

        parser = DocumentParser(use_markitdown=True)
        results = parser.parse(tmp_path, batch_size=2)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, ParsedDocument) for r in results)

    def test_config_update(self):
        """Test configuration updates."""
        parser = DocumentParser()
        config = parser._update_config(
            parse_elements={"tables": False},
            embeddings_provider="cloudflare"
        )
        assert config._parse_elements["tables"] is False
        assert config._embeddings_provider == "cloudflare"

    def test_error_handling(self, tmp_path):
        """Test error handling for invalid files."""
        non_existent = tmp_path / "not_exists.pdf"
        parser = DocumentParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse(non_existent)
