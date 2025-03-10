# Contributing to Book Renamer

Thank you for considering contributing to Book Renamer! Here's how you can help:

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/dmitrykrivaltsevich/pdfrename.git
   cd pdfrename
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. Install development dependencies:
   ```bash
   pip install pytest pylint black
   ```

## Running Tests

```bash
python -m unittest test_pdfrename.py
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Format code with Black before submitting

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests to ensure they pass
5. Format code with Black: `black pdfrename.py`
6. Commit your changes: `git commit -am 'Add some amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## Adding New Features

When adding new features:

1. Add appropriate tests
2. Update documentation in README.md
3. Document new command-line options

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.