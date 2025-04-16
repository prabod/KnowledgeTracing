"""
Setup for the knowledge tracing project.
"""

from setuptools import find_packages, setup

setup(
    name="knowledge_tracing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "pandas",
        "numpy",
        "lightning",
        "datasets",
        "transformers",
    ],
)
