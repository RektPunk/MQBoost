from setuptools import setup, find_packages

setup(
    name="quantile-tree",
    version="0.0.5",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
    ],
    author="RektPunk",
    author_email="rektpunk@gmail.com",
    description="Monotone quantile regressor",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RektPunk/quantile-tree",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
