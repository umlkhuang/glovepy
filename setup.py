import os
from setuptools import setup, find_packages

def readfile(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='glovepy',
    version='0.0.1',
    description='Python implementation of the GloVe algorithm',
    long_description=readfile('README.md'),
    ext_modules=[],
    packages=find_packages(),
    py_modules = [],
    author='Ke Huang',
    author_email='khuang at cs uml edu',
    url='https://github.com/umlkhuang/glovepy',
    download_url='https://github.com/umlkhuang/glovepy',
    keywords='NLP, Machine Learning',
    license='MIT',
    platforms='any',
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6'
    ],
    setup_requires = [],
    install_requires=[
        'cython',
        'numpy'
    ],
    include_package_data=True,
)
