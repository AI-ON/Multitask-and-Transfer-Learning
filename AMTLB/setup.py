from setuptools import setup, find_packages

VERSION = 0.1
README = open('README.md')

setup(
    name="amtlb",
    version=VERSION,
    description="Atari Multitask and Transfer Learning Benchmark",
    author="Josh Kuhn",
    author_email="deontologician@gmail.com",
    url="ai-on.org/projects/multitask-and-transfer-learning.html",
    packages=find_packages('amtlb'),
    package_dir={'': 'amtlb'},
    license="Apache License",
    tests_require=[
        'pytest == 3.0.6',
        'mock'
    ],
    long_description=README,
    install_requires=[
        'gym >= 0.5.6',
    ],
    scripts=[
        'runtests',
    ],
)
