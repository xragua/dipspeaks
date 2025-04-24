from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="dipspeaks",
    version="0.1",
    author="LAEX",
    author_email="graciela.sanjurjo@ua.es",
    description="A python package to detect relevant peaks and dips in lightcurves.",
    url="https://github.com/xragua/dipspeaks/",
    packages=find_packages(),
    python_requires=">=3.8",

    # NEW — modern licensing metadata
    license="MIT",
    license_files=("LICENSE",),          # <- note the *plural* key

    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "tensorflow",
        "scikit-learn",
    ],
)
