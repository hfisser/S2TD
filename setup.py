from setuptools import setup

setup(
    name="s2td",
    install_requires=[
        "numpy",
        "pandas",
        "geopandas",
        "xarray",
        "fiona",
        "rasterio",
        "scikit-learn",
        "osmpythontools"
    ],
)