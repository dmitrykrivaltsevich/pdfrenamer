# Book Renamer

A Python utility that automatically renames book files (PDF, EPUB) based on their metadata, organizing them in a consistent format: `YEAR - AUTHOR. TITLE.YEAR.pdf`

## Features

- Multiple file format support:
  - PDF files with full metadata extraction
  - EPUB files with metadata extraction
  - Extensible to support other formats via command-line
- Extracts ISBN numbers from PDF content (supports both ISBN-10 and ISBN-13)
- OCR support for scanned documents using Tesseract (with Russian language support)
- Fetches book metadata from multiple free sources:
  - Google Books API
  - Open Library API
  - WorldCat (web scraping)
  - Russian State Library (RSL) catalog (for Russian books)
  - eLibrary.ru (for Russian academic books)
  - Anna's Archive (comprehensive catalog for many languages)
  - Built-in PDF/EPUB metadata
- Recursive directory scanning for batch processing
- Custom naming templates via command-line options
- Duplicate detection via file hash checking
- Proper Unicode and Cyrillic character support
- Multithreading for faster processing
- File-based logging
- Automatic backup of original files
- Metadata caching for faster processing

## Installation

### Standard Installation

```bash
pip install -r requirements.txt
```

### Using uv (Fast Python Package Installer)

[uv](https://github.com/astral-sh/uv) is a much faster alternative to pip:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies using uv
uv pip install -r requirements.txt
```

### Using pipx (Isolated Environment)

[pipx](https://github.com/pypa/pipx) lets you install and run the tool in an isolated environment:

```bash
# Install pipx if you don't have it
python -m pip install --user pipx
python -m pipx ensurepath

# Install the tool in an isolated environment
cd /path/to/repo
pipx install .

# Run the tool from anywhere
pdfrename --source-dir /path/to/books --destination-dir ~/Books
```

### Create an Executable with PyInstaller

For a standalone executable with no Python dependencies:

```bash
pip install pyinstaller
pyinstaller --onefile pdfrename.py

# The executable will be in the dist/ directory
./dist/pdfrename --source-dir /path/to/books
```

Required dependencies:
- PyMuPDF (fitz)
- PyPDF2
- requests

Optional dependencies for specific formats:
- EPUB support:
  - ebooklib

Optional dependencies for OCR support:
- pytesseract
- pdf2image
- Pillow
- System dependencies (not installable via pip):
  - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
    - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
    - macOS: `brew install tesseract`
    - Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
  - Poppler: Required by pdf2image for PDF to image conversion
    - Ubuntu/Debian: `sudo apt-get install poppler-utils`
    - macOS: `brew install poppler`
    - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases

### Docker Installation

For a containerized installation with all dependencies:

```bash
# Build the Docker image
docker build -t book-renamer .

# Run the tool
docker run -v /path/to/books:/books -v /path/to/output:/output book-renamer \
  --source-dir /books --destination-dir /output
```

## Usage

```bash
python pdfrename.py --source-dir /path/to/books --destination-dir /path/to/output [OPTIONS]
```

### Arguments

#### Basic Options:
- `--source-dir`: Directory containing book files to process (required)
- `--destination-dir`: Output directory for renamed files (default: ~/Uploads)
- `--dry-run`: Test run without actually renaming files
- `--recursive`: Process subdirectories recursively

#### File Format Options:
- `--include-epub`: Include EPUB files in processing (defaults to PDF only)
- `--extensions`: Additional file extensions to process, comma-separated (e.g. ".mobi,.azw")

#### Advanced Options:
- `--format`: Custom filename format template (default: {year} - {author}. {title}.{year}.pdf)
- `--log-level`: Set logging level (debug, info, warning, error, critical)
- `--log-file`: Write logs to this file
- `--backup-dir`: Create backups of original files in this directory
- `--threads`: Number of concurrent threads for processing (default: 4)
- `--cache-file`: Metadata cache file location (default: .pdf_metadata_cache.json)
- `--transliterate-cyrillic`: Transliterate Cyrillic characters to Latin for maximum compatibility

#### OCR Options:
- `--disable-ocr`: Disable OCR for scanned PDFs (makes processing faster but less thorough)
- `--ocr-dpi`: DPI for OCR image processing (default: 300). Higher values give better results but are slower
- `--ocr-max-pages`: Maximum number of pages to OCR per document (default: 5)
- `--ocr-lang`: OCR language for text extraction (default: eng). Use 'rus' for Russian or 'rus+eng' for mixed content

> **Note**: For languages other than English, install the appropriate Tesseract language packages:
> - Ubuntu/Debian: `sudo apt-get install tesseract-ocr-rus tesseract-ocr-deu tesseract-ocr-fra`
> - macOS: `brew install tesseract-lang`
> - Windows: Select additional languages during installation
>
> For Russian books, use `--ocr-lang rus` or `--ocr-lang rus+eng` for mixed content

### Format Templates

You can customize the output filename format using these template variables:
- `{author}`: Book author(s)
- `{title}`: Book title
- `{year}`: Publication year
- `{isbn}`: ISBN number (if found)

Example: `--format "{year} - {author} - {title} [{isbn}].pdf"`

### How It Works
1. Identifies supported book file formats (.pdf, .epub, or other specified formats)
2. For PDFs:
   - Scans PDF content for valid ISBN numbers
   - For scanned documents, uses OCR to extract text and find ISBN numbers
3. For EPUBs:
   - Extracts metadata from the EPUB container
4. Looks up book metadata using multiple free APIs (when ISBN available)
5. Falls back to embedded document metadata when online lookup fails
6. Renames files using specified format template

## Testing

Run the unit tests:

```bash
python3 -m unittest test_pdfrename.py
```

## Examples

### PDF Example
Input file: `random_name.pdf`
Output file: `2020 - John Smith. Introduction to Programming.2020.pdf`

### EPUB Example
Input file: `book123.epub`
Output file: `2021 - Jane Doe. Advanced Machine Learning.2021.epub`

### Russian Book Example
Input file: `книга.pdf` (with ISBN 978-5-93700-104-7)
Output file: `2018 - Иванов И.И. Программирование на Python.2018.pdf`

## Russian Book Support

The tool has special support for Russian language books:

1. Specialized metadata sources for Russian ISBNs (books with 978-5-xxx pattern):
   - Russian State Library (RSL) catalog
   - eLibrary.ru for academic books
   - Anna's Archive (provides excellent coverage for Russian literature)

2. OCR support for Russian text:
   - Use `--ocr-lang rus` for Russian-only documents
   - Use `--ocr-lang rus+eng` for mixed Russian/English content
   - Install the Russian language pack for Tesseract:
     - Ubuntu/Debian: `sudo apt-get install tesseract-ocr-rus`
     - macOS: `brew install tesseract-lang`
     - Windows: Select Russian during Tesseract installation

3. Proper Cyrillic filename handling:
   - By default, Cyrillic characters are preserved in filenames
   - Use `--transliterate-cyrillic` if your system has issues with Cyrillic filenames

## License

MIT License