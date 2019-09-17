import os
from setuptools import setup, find_packages

import deepPavlovEval

__location__ = os.path.realpath(os.path.join(os.getcwd(),
                                             os.path.dirname(__file__)))


def read_requirements():
    """parses requirements from requirements.txt"""
    reqs_path = os.path.join(__location__, 'requirements.txt')
    with open(reqs_path, encoding='utf8') as f:
        reqs = [line.strip() for line in f if not line.strip().startswith('#')]

    names = []
    links = []
    for req in reqs:
        if '://' in req:
            links.append(req)
        else:
            names.append(req)
    return {'install_requires': names, 'dependency_links': links}


def readme():
    with open(os.path.join(__location__, 'README.md'), encoding='utf8') as f:
        text = f.read()
    return text


setup(
    name="deepPavlovEval",
    version=deepPavlovEval.__version__,
    description=deepPavlovEval.__description__,
    long_description=readme(),
    long_description_content_type="text/markdown",
    author=deepPavlovEval.__author__,
    author_email=deepPavlovEval.__email__,
    url="https://github.com/deepmipt/deepPavlovEval",
    license=deepPavlovEval.__license__,
    packages=find_packages(),
    include_package_data=True,
    keywords=deepPavlovEval.__keywords__,
    **read_requirements()
)
