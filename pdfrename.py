#!/usr/bin/env python3

import re
import shutil
import glob
import argparse
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
import logging
import requests
import os
import hashlib
import json
import concurrent.futures
import time
from pathlib import Path
import threading
import tempfile
import unicodedata
from typing import Dict, List, Tuple, Optional, Any, Set, Union
import zipfile
import io

# Import unidecode for transliteration with fallback
try:
    from unidecode import unidecode

    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

# Import ebooklib for EPUB support with fallback
try:
    import ebooklib
    from ebooklib import epub
    from ebooklib.utils import parse_html_string

    EPUB_SUPPORT = True
except ImportError:
    EPUB_SUPPORT = False

# Import OCR-related libraries, with fallback if not available
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configure logging
DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Constants
ISBN13_REGEX = r"97[89]-?\d{1,5}-?\d{1,7}-?\d{1,6}-?\d"
ISBN10_REGEX = r"\d{1,5}-?\d{1,7}-?\d{1,6}-[\dxX]"
DEFAULT_FORMAT = "{year} - {author}. {title}.{year}.pdf"
DEFAULT_DEST_DIR = os.path.expanduser("~/Uploads")
METADATA_CACHE_FILE = ".pdf_metadata_cache.json"
OCR_MAX_PAGES = 5  # Maximum number of pages to OCR for ISBN extraction
OCR_DPI = 300  # DPI for image conversion - higher values give better OCR results but are slower

# Global variables
metadata_cache = {}
lock = threading.Lock()
processed_hashes: Set[str] = set()


def setup_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """Configure logging based on user preferences."""
    level = LOG_LEVELS.get(log_level.lower(), logging.INFO)

    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())

    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT, handlers=handlers)


def load_metadata_cache(cache_path: str) -> Dict:
    """Load the metadata cache from disk."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load metadata cache: {e}")
    return {}


def save_metadata_cache(cache: Dict, cache_path: str) -> None:
    """Save the metadata cache to disk."""
    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
    except IOError as e:
        logging.error(f"Failed to save metadata cache: {e}")


def is_valid_isbn13(isbn: str) -> bool:
    """Validate ISBN-13 number using checksum algorithm."""
    if len(isbn) != 13:
        return False
    multiplier_isbn13 = [1, 3]
    check = sum(multiplier_isbn13[i % 2] * int(x) for i, x in enumerate(isbn[:-1]))
    return (10 - check % 10) % 10 == int(isbn[-1])


def is_valid_isbn10(isbn: str) -> bool:
    """Validate ISBN-10 number using checksum algorithm."""
    if len(isbn) != 10:
        return False
    multiplier_isbn10 = list(range(1, 11))
    check = sum(
        multiplier_isbn10[i] * (10 if x.upper() == "X" else int(x))
        for i, x in enumerate(isbn[:-1])
    )
    return check % 11 == (10 if isbn[-1].upper() == "X" else int(isbn[-1]))


def extract_isbn(text: str, isbn_regex: str, validation_func) -> Optional[str]:
    """Extract ISBN from text using regex and validate it."""
    isbn = re.search(isbn_regex, text)
    if isbn:
        isbn = isbn.group(0).replace("-", "")
        if validation_func(isbn):
            return isbn
    return None


def get_book_details_from_google(
    isbn: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch book details from Google Books API using ISBN."""
    assert isbn

    cache_key = f"google_{isbn}"
    if cache_key in metadata_cache:
        logging.debug(f"Using cached data for ISBN {isbn} from Google Books")
        return metadata_cache[cache_key]

    try:
        response = requests.get(
            f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}", timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("totalItems", 0):
            return None, None, None

        book = data["items"][0]["volumeInfo"]
        authors = ", ".join(book.get("authors", []))
        title = book.get("title")
        published_date = book.get("publishedDate")
        year = published_date.split("-")[0] if published_date else None

        result = (authors, title, year)

        # Cache the result
        with lock:
            metadata_cache[cache_key] = result

        return result
    except (requests.RequestException, KeyError, IndexError) as e:
        logging.error(f"Error fetching book details from Google Books: {e}")
        return None, None, None


def get_book_details_from_open_library(
    isbn: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch book details from Open Library API using ISBN."""
    cache_key = f"openlibrary_{isbn}"
    if cache_key in metadata_cache:
        logging.debug(f"Using cached data for ISBN {isbn} from Open Library")
        return metadata_cache[cache_key]

    try:
        response = requests.get(
            f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data",
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        book_key = f"ISBN:{isbn}"
        if book_key in data:
            book = data[book_key]
            authors = ", ".join(author["name"] for author in book.get("authors", []))
            title = book.get("title")
            year = book.get("publish_date", "")
            if year and len(year) >= 4:
                # Extract just the year from various date formats
                year_match = re.search(r"\d{4}", year)
                year = year_match.group(0) if year_match else year[:4]

            result = (authors, title, year)

            # Cache the result
            with lock:
                metadata_cache[cache_key] = result

            return result
        return None, None, None
    except (requests.RequestException, KeyError, IndexError) as e:
        logging.error(f"Error fetching book details from Open Library: {e}")
        return None, None, None


def get_book_details_from_worldcat(
    isbn: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Fetch book details from WorldCat public API (no key required)."""
    cache_key = f"worldcat_{isbn}"
    if cache_key in metadata_cache:
        logging.debug(f"Using cached data for ISBN {isbn} from WorldCat")
        return metadata_cache[cache_key]

    try:
        # Using WorldCat's public data service
        response = requests.get(
            f"https://www.worldcat.org/isbn/{isbn}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )

        if response.status_code != 200:
            return None, None, None

        # Extract data using regex from HTML response
        # This is a simple way to get data without requiring a key
        html = response.text

        author_match = re.search(
            r'<h4 class="dot-separator">\s*<a[^>]*>(.*?)</a>', html
        )
        title_match = re.search(r'property="name">(.*?)</span>', html)
        year_match = re.search(r'class="year">\s*(\d{4})\s*</span>', html)

        author = author_match.group(1) if author_match else None
        title = title_match.group(1) if title_match else None
        year = year_match.group(1) if year_match else None

        result = (author, title, year)

        # Cache the result
        with lock:
            metadata_cache[cache_key] = result

        return result
    except (requests.RequestException, AttributeError) as e:
        logging.debug(f"Error fetching book details from WorldCat: {e}")
        return None, None, None


def get_book_details_from_epub(
    epub_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract book metadata from EPUB file."""
    if not EPUB_SUPPORT:
        logging.warning(
            "EPUB support is not available. Install ebooklib to enable EPUB support."
        )
        return None, None, None

    try:
        book = epub.read_epub(epub_path)

        # Extract metadata
        title = book.get_metadata("DC", "title")
        title = title[0][0] if title else None

        authors = book.get_metadata("DC", "creator")
        author = ", ".join([a[0] for a in authors]) if authors else None

        # Try to find publication date/year
        dates = book.get_metadata("DC", "date")
        year = None
        if dates:
            for date_info in dates:
                date_str = date_info[0]
                # Try to extract year from various date formats
                year_match = re.search(r"\b(\d{4})\b", date_str)
                if year_match:
                    year = year_match.group(1)
                    break

        return author, title, year
    except Exception as e:
        logging.error(f"Error extracting metadata from EPUB: {e}")

        # Fallback method using zipfile
        try:
            with zipfile.ZipFile(epub_path, "r") as epub_zip:
                # Try to find and parse content.opf
                for filename in epub_zip.namelist():
                    if filename.endswith(".opf"):
                        content = epub_zip.read(filename).decode("utf-8")

                        # Extract simple metadata using regex
                        title_match = re.search(
                            r"<dc:title[^>]*>(.*?)</dc:title>", content
                        )
                        title = title_match.group(1) if title_match else None

                        author_match = re.search(
                            r"<dc:creator[^>]*>(.*?)</dc:creator>", content
                        )
                        author = author_match.group(1) if author_match else None

                        date_match = re.search(
                            r"<dc:date[^>]*>(.*?)</dc:date>", content
                        )
                        if date_match and date_match.group(1):
                            year_match = re.search(r"\b(\d{4})\b", date_match.group(1))
                            year = year_match.group(1) if year_match else None
                        else:
                            year = None

                        return author, title, year

        except Exception as inner_e:
            logging.error(f"Error with fallback EPUB extraction: {inner_e}")

    return None, None, None


def get_book_details_from_pdf(
    pdf_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract book metadata directly from PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            info = reader.metadata
            if info is not None:
                author = info.get("/Author")
                title = info.get("/Title")
                # Try multiple date fields
                year = None
                for date_field in ["/CreationDate", "/ModDate"]:
                    date = info.get(date_field)
                    if date and len(date) >= 6:
                        # Extract year from PDF date format (D:YYYYMMDD)
                        year_match = re.search(r"D:(\d{4})", date)
                        if year_match:
                            year = year_match.group(1)
                            break

                return author, title, year
    except Exception as e:
        logging.error(f"Error reading PDF with PyPDF2: {e}. Trying with PyMuPDF.")

    try:
        doc = fitz.open(pdf_path)
        info = doc.metadata
        if info is not None:
            author, title = info.get("author"), info.get("title")

            # Try to extract year from various date fields
            year = None
            for date_field in ["creationDate", "modDate"]:
                date = info.get(date_field)
                if date and len(date) >= 6:
                    year_match = re.search(r"D:(\d{4})", date)
                    if year_match:
                        year = year_match.group(1)
                        break

            return author, title, year
    except Exception as e:
        logging.error(f"Error reading PDF with PyMuPDF: {e}")

    return None, None, None


def get_book_details_from_file(
    file_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract book metadata from a file based on its extension."""
    _, ext = os.path.splitext(file_path.lower())

    if ext == ".pdf":
        return get_book_details_from_pdf(file_path)
    elif ext == ".epub":
        return get_book_details_from_epub(file_path)
    else:
        logging.warning(f"Unsupported file format: {ext}")
        return None, None, None


def perform_ocr_on_pdf(pdf_path: str, max_pages: int = OCR_MAX_PAGES) -> List[str]:
    """
    Perform OCR on PDF pages to extract text from scanned documents.
    Returns a list of extracted text from each page.
    """
    if not OCR_AVAILABLE:
        logging.warning(
            "OCR libraries not available. Install pytesseract, pdf2image and Pillow for OCR support."
        )
        logging.warning("pip install pytesseract pdf2image pillow")
        logging.warning("You also need to install these system dependencies:")
        logging.warning("1. Tesseract OCR:")
        logging.warning("  - Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        logging.warning("  - macOS: brew install tesseract")
        logging.warning(
            "  - Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki"
        )
        logging.warning("2. Poppler (required by pdf2image):")
        logging.warning("  - Ubuntu/Debian: sudo apt-get install poppler-utils")
        logging.warning("  - macOS: brew install poppler")
        logging.warning(
            "  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases"
        )
        logging.warning("    and add the bin directory to your PATH")
        return []

    try:
        logging.info(f"Attempting OCR on {pdf_path} (max {max_pages} pages)...")
        extracted_texts = []

        # Create a temporary directory for image conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Convert PDF pages to images
                images = convert_from_path(
                    pdf_path,
                    dpi=OCR_DPI,
                    first_page=1,
                    last_page=max_pages,
                    output_folder=temp_dir,
                )
            except Exception as e:
                if "Unable to get page count" in str(e) or "poppler" in str(e).lower():
                    logging.error(
                        "PDF to image conversion failed. Poppler is not installed or not in PATH."
                    )
                    logging.error("To install Poppler:")
                    logging.error(
                        "  - Ubuntu/Debian: sudo apt-get install poppler-utils"
                    )
                    logging.error("  - macOS: brew install poppler")
                    logging.error(
                        "  - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases"
                    )
                    logging.error("    and add the bin directory to your PATH")
                else:
                    logging.error(f"Error converting PDF to images: {e}")
                return []

            if not images:
                logging.warning(f"No images extracted from PDF {pdf_path}")
                return []

            # Process each image with OCR
            for i, image in enumerate(images):
                try:
                    logging.debug(f"Performing OCR on page {i+1}...")
                    text = pytesseract.image_to_string(image)
                    extracted_texts.append(text)
                except Exception as e:
                    if "tesseract" in str(e).lower():
                        logging.error(
                            "Tesseract OCR failed. Make sure Tesseract is installed and in PATH."
                        )
                        logging.error("To install Tesseract OCR:")
                        logging.error(
                            "  - Ubuntu/Debian: sudo apt-get install tesseract-ocr"
                        )
                        logging.error("  - macOS: brew install tesseract")
                        logging.error(
                            "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
                        )
                    else:
                        logging.error(f"Error performing OCR on page {i+1}: {e}")
                    # Continue with other pages

        return extracted_texts
    except Exception as e:
        logging.error(f"Error during OCR processing: {e}")
        return []


def find_isbn_in_pdf(pdf_path: str) -> Optional[str]:
    """Search for ISBN numbers in PDF content, using OCR if regular extraction fails."""
    limit_search_first_pages = 10

    # First try normal text extraction methods
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages[:limit_search_first_pages]:
                text = page.extract_text()
                if text and len(text.strip()) > 0:  # Only process if we got some text
                    isbn = extract_isbn(text, ISBN13_REGEX, is_valid_isbn13)
                    if isbn:
                        return isbn
                    isbn = extract_isbn(text, ISBN10_REGEX, is_valid_isbn10)
                    if isbn:
                        return isbn
    except Exception as e:
        logging.error(f"Error reading PDF with PyPDF2: {e}. Trying with PyMuPDF.")

    # Try with PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        has_text = False
        for page in doc[:limit_search_first_pages]:
            text = page.get_text()
            if text and len(text.strip()) > 0:
                has_text = True
                isbn = extract_isbn(text, ISBN13_REGEX, is_valid_isbn13)
                if isbn:
                    return isbn
                isbn = extract_isbn(text, ISBN10_REGEX, is_valid_isbn10)
                if isbn:
                    return isbn

        # If no text was extracted or no ISBN found, this might be a scanned PDF
        if not has_text:
            logging.info(
                f"No readable text found in {pdf_path}. This might be a scanned document."
            )
    except Exception as e:
        logging.error(f"Error reading PDF with PyMuPDF: {e}")

    # If standard extraction failed, try OCR if available
    if OCR_AVAILABLE:
        logging.info("Attempting to extract text using OCR...")
        ocr_texts = perform_ocr_on_pdf(pdf_path, OCR_MAX_PAGES)

        for text in ocr_texts:
            if text and len(text.strip()) > 0:
                isbn = extract_isbn(text, ISBN13_REGEX, is_valid_isbn13)
                if isbn:
                    logging.info(f"Found ISBN via OCR: {isbn}")
                    return isbn
                isbn = extract_isbn(text, ISBN10_REGEX, is_valid_isbn10)
                if isbn:
                    logging.info(f"Found ISBN via OCR: {isbn}")
                    return isbn

    return None


def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash for a file to detect duplicates."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except IOError as e:
        logging.error(f"Error calculating file hash: {e}")
        return ""


def safe_filename(text: str, preserve_cyrillic: bool = True) -> str:
    """
    Convert text to safe filename by removing invalid characters and handling Unicode.

    Args:
        text: The text to convert into a safe filename
        preserve_cyrillic: If True, keep Cyrillic characters. If False, transliterate to Latin.
    """
    if not text:
        return "Unknown"

    # Normalize Unicode (NFC form)
    safe = unicodedata.normalize("NFC", text)

    # Optionally transliterate Cyrillic to Latin
    if not preserve_cyrillic and UNIDECODE_AVAILABLE:
        # Detect if text contains Cyrillic
        has_cyrillic = bool(re.search("[\u0400-\u04ff]", safe))
        if has_cyrillic:
            safe = unidecode(safe)

    # Replace invalid filename characters and problematic punctuation
    safe = re.sub(r'[\\/*?:"<>|;]', "-", safe)

    # Clean up any double dashes
    safe = re.sub(r"-{2,}", "-", safe)

    # Remove leading and trailing dashes and whitespace
    safe = safe.strip("-").strip()

    # Limit filename length (safely handling UTF-8 multi-byte characters)
    if len(safe) > 100:
        # Ensure we don't cut in the middle of a UTF-8 character
        safe = safe[:100]
        # Make sure we have a valid UTF-8 string by decoding and re-encoding
        try:
            safe.encode("utf-8").decode("utf-8")
        except UnicodeDecodeError:
            # If we cut in the middle of a multi-byte character,
            # remove the last character until we have valid UTF-8
            while len(safe) > 0:
                safe = safe[:-1]
                try:
                    safe.encode("utf-8").decode("utf-8")
                    break
                except UnicodeDecodeError:
                    continue

    return safe


def format_output_filename(
    template: str, metadata: Dict[str, str], preserve_cyrillic: bool = True
) -> str:
    """Format output filename using the provided template."""
    # Clean up metadata values for safe filenames
    safe_metadata = {
        k: safe_filename(v, preserve_cyrillic) if v else "Unknown"
        for k, v in metadata.items()
    }

    try:
        return template.format(**safe_metadata)
    except KeyError as e:
        logging.error(f"Invalid template key: {e}")
        # Fall back to default format
        return DEFAULT_FORMAT.format(**safe_metadata)


def create_backup(file_path: str, backup_dir: str) -> bool:
    """Create a backup of the original file."""
    if not os.path.exists(backup_dir):
        try:
            os.makedirs(backup_dir)
        except OSError as e:
            logging.error(f"Failed to create backup directory: {e}")
            return False

    try:
        filename = os.path.basename(file_path)
        backup_path = os.path.join(backup_dir, filename)
        # Add timestamp if file already exists
        if os.path.exists(backup_path):
            timestamp = time.strftime("%Y%m%d%H%M%S")
            name, ext = os.path.splitext(filename)
            backup_path = os.path.join(backup_dir, f"{name}_{timestamp}{ext}")

        shutil.copy2(file_path, backup_path)
        logging.info(f"Created backup at {backup_path}")
        return True
    except IOError as e:
        logging.error(f"Failed to create backup: {e}")
        return False


def rename_file(
    filename: str,
    destination_dir: str,
    format_template: str,
    organize_by_genre: bool,
    backup_dir: Optional[str],
    dry_run: bool = False,
    transliterate_cyrillic: bool = False,
) -> bool:
    """Process a book file (PDF or EPUB) and rename it based on metadata."""
    try:
        logging.info(f"Processing {filename} ...")

        # Determine file type by extension
        _, ext = os.path.splitext(filename.lower())
        is_epub = ext == ".epub"
        is_pdf = ext == ".pdf"

        if not (is_epub or is_pdf):
            logging.warning(f"Unsupported file format: {ext}")
            return False

        # Calculate file hash to detect duplicates
        file_hash = calculate_file_hash(filename)
        if file_hash in processed_hashes:
            logging.warning(f"Skipping duplicate file: {filename}")
            return False

        processed_hashes.add(file_hash)

        # Create backup if requested
        if backup_dir and not dry_run:
            if not create_backup(filename, backup_dir):
                logging.warning(f"Proceeding without backup for {filename}")

        # Get metadata based on file type
        isbn = None
        author = title = year = None

        # For PDFs, we can search for ISBN
        if is_pdf:
            isbn = find_isbn_in_pdf(filename)

        if isbn:
            logging.info(f"Found ISBN: {isbn}")
            # Try different APIs in sequence
            author, title, year = get_book_details_from_google(isbn)
            if not all([author, title, year]):
                author, title, year = get_book_details_from_open_library(isbn)
            if not all([author, title, year]):
                author, title, year = get_book_details_from_worldcat(isbn)

        # If we still don't have metadata, try extracting from the file
        if not all([author, title, year]):
            logging.info(
                f"ISBN not found or metadata incomplete. Trying to extract metadata from file."
            )
            file_author, file_title, file_year = get_book_details_from_file(filename)
            author = author or file_author
            title = title or file_title
            year = year or file_year

        if all([author, title, year]):
            # Create metadata dictionary for filename formatting
            metadata = {
                "author": author,
                "title": title.replace(":", "-").strip() if title else "Unknown",
                "year": year,
                "isbn": isbn or "Unknown",
            }

            # Format the filename exactly as specified
            preserve_cyrillic = not transliterate_cyrillic
            new_file_name = format_output_filename(
                format_template, metadata, preserve_cyrillic
            )
            if not new_file_name.endswith(".pdf"):
                new_file_name += ".pdf"

            # Set destination path
            destination = os.path.join(destination_dir, new_file_name)

            # If file already exists, log a warning but don't modify the filename
            if os.path.exists(destination) and not dry_run:
                logging.warning(f"Destination file already exists: {destination}")
                logging.warning("Will overwrite existing file")

            logging.info(
                f'{"Dry run: " if dry_run else ""}Renaming {filename} to {destination}'
            )

            if not dry_run:
                try:
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.move(filename, destination)
                    return True
                except Exception as e:
                    logging.error(f"Error renaming file: {e}")
            else:
                return True  # Consider dry-run a success
        else:
            logging.warning(f"Skipping {filename}: insufficient metadata")
    except Exception as e:
        logging.error(f"Unexpected error processing {filename}: {e}")

    return False


def find_book_files(
    source_dir: str, recursive: bool = False, extensions: List[str] = None
) -> List[str]:
    """
    Find all book files in the given directory.

    Args:
        source_dir: Directory to search in
        recursive: Whether to search recursively in subdirectories
        extensions: List of file extensions to search for (default: ['.pdf', '.epub'])
    """
    # Validate source directory
    if not source_dir:
        raise ValueError("Source directory is required")

    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory does not exist: {source_dir}")

    if not os.path.isdir(source_dir):
        raise ValueError(f"Source path is not a directory: {source_dir}")

    if extensions is None:
        extensions = [".pdf", ".epub"]

    # Normalize extensions to lowercase
    extensions = [
        ext.lower() if not ext.startswith(".") else ext.lower() for ext in extensions
    ]
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

    book_files = []

    if recursive:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_ext = os.path.splitext(file.lower())[1]
                if file_ext in extensions:
                    book_files.append(os.path.join(root, file))
    else:
        for ext in extensions:
            book_files.extend(glob.glob(os.path.join(source_dir, f"*{ext}")))

    return book_files


# For backward compatibility
def find_pdf_files(source_dir: str, recursive: bool = False) -> List[str]:
    """Find all PDF files in the given directory."""
    return find_book_files(source_dir, recursive, [".pdf"])


def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description="PDF Book Renamer - Automatically rename PDF files based on ISBN or metadata"
    )
    parser.add_argument(
        "--source-dir", required=True, help="Folder that contains PDF files (required)"
    )
    parser.add_argument(
        "--destination-dir",
        default=DEFAULT_DEST_DIR,
        help=f"Output directory for renamed files (default: {DEFAULT_DEST_DIR})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument(
        "--recursive", action="store_true", help="Process subdirectories recursively"
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_FORMAT,
        help="Output filename format (default: {year} - {author}. {title}.{year}.pdf)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=LOG_LEVELS.keys(),
        help="Set logging level (default: info)",
    )
    parser.add_argument("--log-file", help="Write logs to this file")
    parser.add_argument(
        "--backup-dir", help="Create backups of original files in this directory"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of concurrent threads for processing (default: 4)",
    )
    parser.add_argument(
        "--cache-file",
        default=METADATA_CACHE_FILE,
        help=f"Metadata cache file location (default: {METADATA_CACHE_FILE})",
    )
    parser.add_argument(
        "--disable-ocr",
        action="store_true",
        help="Disable OCR for scanned PDFs (makes processing faster but less thorough)",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=OCR_DPI,
        help=f"DPI for OCR image processing (default: {OCR_DPI}). Higher values give better results but are slower",
    )
    parser.add_argument(
        "--ocr-max-pages",
        type=int,
        default=OCR_MAX_PAGES,
        help=f"Maximum number of pages to OCR per document (default: {OCR_MAX_PAGES})",
    )
    parser.add_argument(
        "--transliterate-cyrillic",
        action="store_true",
        help="Transliterate Cyrillic characters to Latin (for maximum compatibility)",
    )

    # File format options
    parser.add_argument(
        "--include-epub",
        action="store_true",
        help="Include EPUB files in processing (defaults to PDF only)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        help='Additional file extensions to process, comma-separated (e.g. ".mobi,.azw")',
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level, args.log_file)

    # Display arguments
    logging.info(f"Arguments: {args}")

    # Set OCR parameters from args
    global OCR_AVAILABLE

    # Update the constants with args values
    globals()["OCR_DPI"] = args.ocr_dpi
    globals()["OCR_MAX_PAGES"] = args.ocr_max_pages

    if args.disable_ocr:
        OCR_AVAILABLE = False
        logging.info("OCR processing disabled by user")
    elif OCR_AVAILABLE:
        logging.info(f"OCR enabled (max {OCR_MAX_PAGES} pages at {OCR_DPI} DPI)")
    else:
        logging.warning(
            "OCR libraries not available. Install pytesseract and pdf2image for OCR support."
        )

    # Ensure destination directory exists
    if not os.path.exists(args.destination_dir) and not args.dry_run:
        try:
            os.makedirs(args.destination_dir)
        except OSError as e:
            logging.error(f"Failed to create destination directory: {e}")
            return

    # Load metadata cache
    global metadata_cache
    cache_path = os.path.join(args.destination_dir, args.cache_file)
    metadata_cache = load_metadata_cache(cache_path)

    # Process file extension filters
    extensions = [".pdf"]  # PDF always included by default
    if getattr(args, "include_epub", False):
        extensions.append(".epub")
    if hasattr(args, "extensions") and args.extensions:
        extensions.extend(args.extensions.split(","))

    # Find all book files to process
    try:
        book_files = find_book_files(args.source_dir, args.recursive, extensions)
        logging.info(f"Found {len(book_files)} book files to process")

        # Count by extension type
        ext_counts = {}
        for file in book_files:
            ext = os.path.splitext(file.lower())[1]
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        for ext, count in ext_counts.items():
            logging.info(f"  - {ext} files: {count}")

        if not book_files:
            logging.warning(
                f"No book files found in {args.source_dir}"
                + (" and its subdirectories" if args.recursive else "")
            )
            return
    except ValueError as e:
        logging.error(f"Error finding book files: {e}")
        return

    # Display stats
    success_count = 0

    # Process files in parallel if more than one thread is specified
    if args.threads > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.threads
        ) as executor:
            future_to_file = {
                executor.submit(
                    rename_file,
                    filename,
                    args.destination_dir,
                    args.format,
                    False,  # organize_by_genre always False
                    args.backup_dir,
                    args.dry_run,
                    args.transliterate_cyrillic,
                ): filename
                for filename in book_files
            }

            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")
    else:
        # Process files sequentially
        for filename in book_files:
            if rename_file(
                filename,
                args.destination_dir,
                args.format,
                False,
                args.backup_dir,
                args.dry_run,
                args.transliterate_cyrillic,
            ):
                success_count += 1

    # Save metadata cache
    save_metadata_cache(metadata_cache, cache_path)

    # Display summary
    logging.info(
        f"Processing completed. Successfully processed {success_count} out of {len(book_files)} files."
    )


def run():
    """Entry point for the application script"""
    main()


if __name__ == "__main__":
    run()
