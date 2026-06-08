from setuptools import find_packages, setup


setup(
    name="change_point_tool",
    version="0.1.0",
    description="A Python package for change-point detection",
    python_requires=">=3.6",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy<=1.19.5",
        "gurobipy>=9.1.0",
    ],
)
