from setuptools import find_packages, setup

with open("neural_physics/__init__.py") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break
    else:
        raise ValueError("Could not find version in __init__.py")


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="neural-physics",
    version=version,
    description="",
    packages=find_packages(),
    install_requires=requirements,
)
