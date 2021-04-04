import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="oleda", # Replace with your own username
    version="0.0.3",
    author="Banuba",
    author_email="olga.matusevich@banuba.com",
    description="an eda",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Banuba/oleda",
    project_urls={
        "Bug Tracker": "https://github.com/Banuba/oleda/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6",
)
