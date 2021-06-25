from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="forge",
    version="0.0.1",
    description="Components and utilities for Qiskit-Metal",
    long_description=readme,
    author=["Ashutosh Bhudia"],
    author_email=[
        "ashu.bhudia@gmail.com",
        ]
    url="https://github.com/ambhudia/forge,
    license=license,
    packages=find_packages(),
)
