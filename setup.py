from setuptools import setup, find_packages

setup(
    name="dipspeaks",
    version="0.1",
    author="LAEX",
    author_email="graciela.sanjurjo@ua.es",
    description="A python package to detect relevant peaks and dips in lightcurves.",
    packages=find_packages(),  # Automatically find packages in dipspeaks/
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "tensorflow",
        "scikit-learn",
    ],
    python_requires=">=3.6",
)

