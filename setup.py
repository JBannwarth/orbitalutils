from setuptools import setup

setup(
    name='orbitutils',
    version='0.1.0',
    author='Jeremie X. J. Bannwarth',
    author_email='jban039@aucklanduni.ac.nz',
    packages=['orbitutils', 'orbitutils.test'],
    # scripts=['bin/script1','bin/script2'],
    url='https://github.com/JBannwarth/orbitutils',
    license='LICENSE',
    description='Utilities to solve orbital mechanics problems, based on ' + \
        'Orbital Mechanics for Engineering Students (4th Edition) by Howard ' + \
        'Curtis',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy",
    ],
)