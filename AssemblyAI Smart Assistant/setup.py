from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ["absl-py==0.12.0"]




setup(
    name = "image_generation",
    version = "0.02",
    install_requires = REQUIRED_PACKAGES,
    packages = find_packages(),
    include_package_data = True,
    description = "Image gen app package"
)