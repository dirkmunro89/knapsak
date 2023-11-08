from setuptools import setup, find_packages

_py3_packages = find_packages(include=['./'])

setup(
    name="Python 3",
    version="0.0.1",
    description='General environment for Python 3',
    author_email='dirkmunro8@gmail.com',
    packages=_py3_packages,
    keywords=['Python 3'],
    python_requires='>=3.8, <4',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'cvxopt',
        'pytest',
        'joblib',
        'vtk'
    ]
)
