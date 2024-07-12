from setuptools import find_packages, setup

setup(
    name="mqboost",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "lightgbm>=4.0.0",
        "xgboost>=2.0.0",
    ],
    author="RektPunk",
    author_email="rektpunk@gmail.com",
    description="Monotonic composite quantile gradient boost regressor",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/RektPunk/mqboost",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
