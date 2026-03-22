from setuptools import find_namespace_packages, setup


setup(
    name="k-servers",
    version="0.1.0",
    description="k-server library for WF contexts, graph exploration, and potential evaluation",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "tqdm",
    ],
    python_requires=">=3.10",
)
