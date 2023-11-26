# KNAPSAK


This repository contains a simple Python implementation of a solution procedure for variants of (what is sometimes referred to as) the Orthogonal Minimum Container Packing problem [[1]](#1).

The implementation captured here is intended to form a basis for study of cutting-stock, knapsack, and bin packing problems, with applications in scheduling and production planning for 3D printing.

__The Orthogonal Minimum Container Packing problem__

Given a collection of objects, find the translation and rotation which has to be applied to each object, such that the volume of the axis-aligned bounding box of all the objects is minimized. 

Cite Fogelman

We take the sum of the volumes of the axis-aligned bounding boxes of all the objects, individually, as a reference volume by which the objective is normalised. This is of course the smallest volume container that can possibily be achieved if all the objects are axis-aligned cuboids (of particular sizes), or modelled as such.

## Setup

We require Python 3 with [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), and [VTK](https://vtk.org/) packages installed. 

A virtual environment may be convenient for this. On a Linux system, create a virtual environment `env` and install the required packages with the following commands:
```
$ python3 -m venv ./env
$ source env/bin/activate
$ pip install -r requirements.txt
```
Or on a Windows system, using Powershell: 
```
> python3 -m venv .\env
> .\env\Sctripts\Activate.ps1
> pip install -r .\requirements.txt
```

## References
<a id="1">[1]</a>
Alt, H., & Scharf, N. (2018). 
Approximating smallest containers for packing three-dimensional convex objects. International Journal of Computational Geometry & Applications, 28(02), 111-128.
