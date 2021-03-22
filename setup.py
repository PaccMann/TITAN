"""Package installer."""
import codecs
import os

from setuptools import find_packages, setup

LONG_DESCRIPTION = ''
if os.path.exists('README.md'):
    with open('README.md') as fp:
        LONG_DESCRIPTION = fp.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


setup(
    name='paccmann_tcr',
    version=get_version('paccmann_tcr/__init__.py'),
    description='PyTorch implementations T Cell Receptor binding',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Jannis Born, Anna Weber',
    author_email=('jab@zurich.ibm.com, '
                  'wbr@zurich.ibm.com'),
    url='https://github.com/PaccMann/TITAN',
    license='MIT',
    install_requires=[
        'numpy', 'torch>=1.0.0', 'scipy', 'scikit-learn', 'pandas',
        'matplotlib', 'seaborn', 'python-Levenshtein',
        'paccmann_predictor @ git+https://github.com/PaccMann/paccmann_predictor@0.0.4'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages('.')
)
