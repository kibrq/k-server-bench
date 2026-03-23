from pathlib import Path

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = [
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]


setup(
    name="k-servers",
    version="0.1.0",
    description="k-server library for WF contexts, graph exploration, and potential evaluation",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    install_requires=REQUIREMENTS,
    python_requires=">=3.10",
)
