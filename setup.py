from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Runtime dependencies are installed from requirements.txt by
# scripts/setup/setup.sh, which pins CUDA-specific builds and pulls MetaWorld
# from git; install_requires stays empty so those pins are not re-resolved here.
setup(
    name='fedguide',
    version="0.0.1",
    description=('Diffusion Prior Alignment and Value Baseline Guidance for '
                 'Heterogeneous Federated Reinforcement Learning (CoRL 2026)'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Zhilin He, Gauri Joshi',
    author_email='hectorh@andrew.cmu.edu',
    url='https://github.com/hhhhzl/fedguide',
    packages=find_packages(include=['fedguide', 'fedguide.*']),
    python_requires='>=3.10',
    install_requires=[],
)
