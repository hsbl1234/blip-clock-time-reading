#!/usr/bin/env python3

from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="blip-clock-time-reading",
    version="1.0.0",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="Fine-tuned BLIP model for accurate time reading from analog clock images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hsbl1234/blip-clock-time-reading",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "api": [
            "fastapi>=0.75.0",
            "uvicorn>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "blip-clock-predict=examples.predict_single_image:main",
        ],
    },
    keywords="blip, clock, time, reading, computer vision, deep learning, pytorch, transformers",
    project_urls={
        "Bug Reports": "https://github.com/hsbl1234/blip-clock-time-reading/issues",
        "Source": "https://github.com/hsbl1234/blip-clock-time-reading",
        "Documentation": "https://github.com/hsbl1234/blip-clock-time-reading/blob/main/README.md",
    },
)
