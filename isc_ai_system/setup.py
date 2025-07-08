from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="isc-ai-system",
    version="0.1.0",
    author="ISC AI Development Team",
    description="An interactive AI system based on Informational Substrate Convergence hypothesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/isc-ai-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "networkx>=3.0",
        "matplotlib>=3.5.0",
        "colorama>=0.4.6",
        "rich>=13.0.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.8",
        "transformers>=4.30.0",
        "sqlalchemy>=2.0.0",
        "click>=8.0.0",
        "tqdm>=4.65.0",
        "pandas>=1.5.0",
        "plotext>=5.2.0",
    ],
    entry_points={
        "console_scripts": [
            "isc-ai=isc_ai.cli:main",
        ],
    },
)