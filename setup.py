from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()]

setup(
    name="pdfrename",
    version="0.1.0",
    author="Dmitry Krivaltsevich",
    author_email="pdfrename@gomailme.org",
    description="A utility for renaming PDF and EPUB files based on metadata",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dmitrykrivaltsevich/pdfrename",
    py_modules=["pdfrename"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pdfrename=pdfrename:run",
        ],
    },
)