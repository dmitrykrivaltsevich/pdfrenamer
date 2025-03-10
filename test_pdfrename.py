import unittest
from unittest.mock import patch, Mock, MagicMock
import requests
import re
from pdfrename import (
    get_book_details_from_google,
    get_book_details_from_open_library,
    get_book_details_from_pdf,
    find_isbn_in_pdf,
    is_valid_isbn13,
    is_valid_isbn10,
    safe_filename,
)


class TestPdfRename(unittest.TestCase):

    def test_valid_isbn13(self):
        self.assertTrue(is_valid_isbn13("9780306406157"))
        self.assertFalse(is_valid_isbn13("9780306406158"))

    def test_valid_isbn10(self):
        self.assertTrue(is_valid_isbn10("0306406152"))
        self.assertTrue(is_valid_isbn10("044657922X"))
        self.assertFalse(is_valid_isbn10("0306406153"))

    @patch("pdfrename.requests.get")
    def test_get_book_details_from_open_library_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "ISBN:9781234567897": {
                "title": "Test Open Library Book",
                "authors": [{"name": "Author One"}, {"name": "Author Two"}],
                "publish_date": "2020",
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        authors, title, year = get_book_details_from_open_library("9781234567897")
        self.assertEqual(authors, "Author One, Author Two")
        self.assertEqual(title, "Test Open Library Book")
        self.assertEqual(year, "2020")

    @patch("pdfrename.requests.get")
    @patch("pdfrename.logging.error")  # Patch logging.error
    def test_get_book_details_from_open_library_no_book(self, mock_log_error, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        authors, title, year = get_book_details_from_open_library("9781234567897")
        self.assertIsNone(authors)
        self.assertIsNone(title)
        self.assertIsNone(year)

    @patch("pdfrename.requests.get")
    @patch("pdfrename.logging.error")  # Patch logging.error
    def test_get_book_details_from_open_library_request_exception(
        self, mock_log_error, mock_get
    ):
        mock_get.side_effect = requests.RequestException

        authors, title, year = get_book_details_from_open_library("9781234567897")
        self.assertIsNone(authors)
        self.assertIsNone(title)
        self.assertIsNone(year)

    @patch("pdfrename.requests.get")
    def test_get_book_details_from_google_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "totalItems": 1,
            "items": [
                {
                    "volumeInfo": {
                        "title": "Test Google Book",
                        "authors": ["Author One", "Author Two"],
                        "publishedDate": "2020-01-01",
                    }
                }
            ],
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        authors, title, year = get_book_details_from_google("9781234567897")
        self.assertEqual(authors, "Author One, Author Two")
        self.assertEqual(title, "Test Google Book")
        self.assertEqual(year, "2020")

    @patch("pdfrename.requests.get")
    @patch("pdfrename.logging.error")  # Patch logging.error
    def test_get_book_details_from_google_no_book(self, mock_log_error, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {"totalItems": 0}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        authors, title, year = get_book_details_from_google("9781234567897")
        self.assertIsNone(authors)
        self.assertIsNone(title)
        self.assertIsNone(year)

    @patch("pdfrename.requests.get")
    @patch("pdfrename.logging.error")  # Patch logging.error
    def test_get_book_details_from_google_request_exception(
        self, mock_log_error, mock_get
    ):
        mock_get.side_effect = requests.RequestException

        authors, title, year = get_book_details_from_google("9781234567897")
        self.assertIsNone(authors)
        self.assertIsNone(title)
        self.assertIsNone(year)

    @patch("pdfrename.PdfReader")
    def test_get_book_details_from_pdf_success(self, mock_pdfreader):
        mock_reader = Mock()
        mock_reader.metadata = {
            "/Author": "PDF Author",
            "/Title": "PDF Title",
            "/CreationDate": "D:20200101",
        }
        mock_pdfreader.return_value = mock_reader

        with patch("builtins.open", unittest.mock.mock_open()):
            authors, title, year = get_book_details_from_pdf("test.pdf")
            self.assertEqual(authors, "PDF Author")
            self.assertEqual(title, "PDF Title")
            self.assertEqual(year, "2020")

    @patch("pdfrename.PdfReader")
    def test_find_isbn_in_pdf(self, mock_pdfreader):
        mock_page = Mock()
        mock_page.extract_text.return_value = (
            "Some text ISBN 978-0-306-40615-7 more text"
        )
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdfreader.return_value = mock_reader

        with patch("builtins.open", unittest.mock.mock_open()):
            isbn = find_isbn_in_pdf("test.pdf")
            self.assertEqual(isbn, "9780306406157")

    def test_safe_filename_handles_problematic_characters(self):
        """Test that problematic characters are properly handled in safe_filename function."""
        test_cases = [
            # Test case, expected result
            ("James Serra;", "James Serra"),
            ("O'Reilly, Tim", "O'Reilly, Tim"),
            ("Author: Name", "Author- Name"),
            ("Multiple -- Dashes", "Multiple - Dashes"),
            ("Trailing dash-", "Trailing dash"),
            ("-Leading dash", "Leading dash"),
            ("Special/Characters*In?Name", "Special-Characters-In-Name"),
            ("Name with <tag> elements", "Name with -tag- elements"),
            ("Name | with pipes", "Name - with pipes"),
        ]

        for test_input, expected in test_cases:
            with self.subTest(test_input=test_input):
                result = safe_filename(test_input)
                self.assertEqual(result, expected)

    def test_safe_filename_handles_cyrillic(self):
        """Test that Cyrillic characters can be handled correctly."""
        test_cases = ["Достоевский, Фёдор", "Война и мир", "Анна Каренина - Толстой"]

        for test_input in test_cases:
            with self.subTest(test_input=test_input):
                # Just verify that the function doesn't crash with Cyrillic
                result = safe_filename(test_input)

                # Verify the string is valid UTF-8
                try:
                    result.encode("utf-8").decode("utf-8")
                    valid_utf8 = True
                except UnicodeError:
                    valid_utf8 = False
                self.assertTrue(valid_utf8, f"Result should be valid UTF-8: {result}")

                # Verify no invalid filename characters
                invalid_chars = r'[\\/*?:"<>|;]'
                self.assertFalse(
                    bool(re.search(invalid_chars, result)),
                    f"Result should not contain invalid filename chars: {result}",
                )

    @patch("pdfrename.OCR_AVAILABLE", True)
    @patch("pdfrename.perform_ocr_on_pdf")
    @patch("pdfrename.PdfReader")
    @patch("pdfrename.fitz.open")
    def test_find_isbn_in_pdf_with_ocr(
        self, mock_fitz_open, mock_pdfreader, mock_perform_ocr
    ):
        # Mock PdfReader to return no text
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        mock_reader = Mock()
        mock_reader.pages = [mock_page]
        mock_pdfreader.return_value = mock_reader

        # Mock PyMuPDF to return no text
        mock_doc = Mock()
        mock_doc_page = Mock()
        mock_doc_page.get_text.return_value = ""
        mock_doc.__getitem__ = lambda self, x: [mock_doc_page]
        mock_fitz_open.return_value = mock_doc

        # Mock OCR to find an ISBN
        mock_perform_ocr.return_value = [
            "Scanned document with ISBN 978-0-306-40615-7 text"
        ]

        with patch("builtins.open", unittest.mock.mock_open()):
            isbn = find_isbn_in_pdf("test.pdf")
            self.assertEqual(isbn, "9780306406157")
            # Check that OCR was called, method signature varies between unittest versions
            self.assertEqual(mock_perform_ocr.call_count, 1)


if __name__ == "__main__":
    unittest.main()
