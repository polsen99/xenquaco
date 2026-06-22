import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

REQUIREMENTS = (HERE / "requirements.txt").read_text()
requirements = REQUIREMENTS.splitlines()

setup(
    name="xenquaco",
    version="0.1.0",
    description="Quality control tools for 10x Xenium spatial transcriptomics experiments",
    author="Paul Olsen",
    author_email="paul.olsen@alleninstitute.org",
    url="https://github.com/polsen99/xenquaco",
    license="LICENSE",
    packages=find_packages(where="."),
    package_data={
        'xenquaco': ['../ilastik_models/*.ilp'],
    },
    include_package_data=True,
    install_requires=requirements,
)
