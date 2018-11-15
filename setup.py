import setuptools
import deepPavlovEval

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepPavlovEval",
    version=deepPavlovEval.__version__,
    description=deepPavlovEval.__description__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=deepPavlovEval.__author__,
    author_email=deepPavlovEval.__email__,
    url="https://github.com/deepmipt/deepPavlovEval",
    license=deepPavlovEval.__license__,
    packages=setuptools.find_packages(),
    keywords=deepPavlovEval.__keywords__
)
