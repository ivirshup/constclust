from pathlib import Path
from setuptools import setup
import versioneer

setup(
    name="constclust",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
            "pytest",
            "scanpydoc",
            "sphinx"
        ]
    },
    packages=["constclust"],
    zip_safe=False
)
