from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='instapy',
    version='0.0.1',
    author="Alexander Presthus",
    author_email="alexapre@uio.no",
    description="A package for applying instagram-like filters to your images.",
    long_description=long_description,
    long_description_content_type="text/markdown"
    packages=['instapy'],
    scripts=['bin/instapy'],
    url="https://github.uio.no/IN3110/IN3110-alexapre/tree/master/assignment4"
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
