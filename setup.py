from setuptools import setup

setup(
    name='pySOT',
    version='0.1.23',
    packages=['pySOT', 'pySOT.test'],
    url='http://pypi.python.org/pypi/pySOT/',
    license='LICENSE.txt',
    author='David Bindel, David Eriksson, Christine Shoemaker',
    author_email='bindel@cornell.edu, dme65@cornell.edu, shoemaker@nus.edu.sg',
    description='Surrogate Optimization Toolbox',
    long_description=open('README.md').read(),
    install_requires=['pyDOE', 'inspyred', 'pyKriging', 'POAP==0.1.24',
                      'py_dempster_shafer', 'subprocess32', 'matlab_wrapper'],
    classifiers=['Programming Language :: Python :: 2.7'],
)
