from pathlib import Path
from setuptools import setup

setup(
    name="constclust",
    version="0.1.0",
    url="https://github.com/ivirshup/ConsistentClusters",
    author="Isaac Virshup",
    author_email="ivirshup@gmail.com",
    python_requires='>=3.6',
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    extras_require={
        "dev": [
            "pytest"
        ]
    },
    packages=["constclust"],
    zip_safe=False
)
