import os
import sys
import subprocess
import numpy
from setuptools import setup, find_packages, Command, Extension


def readfile(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


class Cythonize(Command):
    """
    Compile the extension .pyx files.
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import Cython
        from Cython.Build import cythonize
        cythonize(define_extensions(cythonize=True))


class Clean(Command):
    """
    Clean build files.
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pth = os.path.dirname(os.path.abspath(__file__))

        subprocess.call(['rm', '-rf', os.path.join(pth, 'build')])
        subprocess.call(['rm', '-rf', os.path.join(pth, 'dist')])
        subprocess.call(['rm', '-rf', os.path.join(pth, 'glovepy.egg-info')])
        subprocess.call(['find', pth, '-name', '*.pyc', '-type', 'f', '-delete'])


def define_extensions(cythonize=False):
    compile_args = ['-fopenmp', '-ffast-math']

    # There are problems with illegal ASM instructions
    # when using the Anaconda distribution (at least on OSX).
    # This could be because Anaconda uses its own assembler?
    # To work around this we do not add -march=native if we
    # know we're dealing with Anaconda
    if 'anaconda' not in sys.version.lower():
        compile_args.append('-march=native')
    
    if cythonize:
        glovepy_inner  = 'glovepy/glove_inner.pyx'
        glovepy_corpus = 'glovepy/corpus_cython.pyx'
    else:
        glovepy_inner  = "glovepy/glove_inner.c"
        glovepy_corpus = "glovepy/corpus_cython.cpp"

    return [Extension("glovepy.glove_inner", [glovepy_inner],
                      extra_link_args=["-fopenmp"],
                      extra_compile_args=compile_args),
            Extension("glovepy.corpus_cython", [glovepy_corpus],
                      language='c++',
                      libraries=["stdc++"],
                      extra_link_args=compile_args,
                      extra_compile_args=compile_args)]
    


setup(
    name='glovepy',
    version='0.0.3',
    description='Python implementation of the GloVe algorithm for word embedding',
    long_description=readfile('README.md'),
    packages=find_packages(),
    py_modules = [],
    author='Ke Huang',
    author_email='khuang@cs.uml.edu',
    url='https://github.com/umlkhuang/glovepy',
    download_url='https://github.com/umlkhuang/glovepy',
    keywords='NLP, Machine Learning',
    license='MIT',
    platforms='linux',
    zip_safe=False,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6'
    ],
    setup_requires = [],
    install_requires=[
        'cython',
        'numpy',
        'scipy'
    ],
    include_package_data=True,
    cmdclass={'cythonize': Cythonize, 'clean': Clean},
    include_dirs = [numpy.get_include()],
    ext_modules=define_extensions()
)
