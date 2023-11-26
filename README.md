# KNAPSAK


This repository contains a simple Python implementation of a solution procedure for variants of (what is sometimes referred to as) the Orthogonal Minimum Container Packing problem [[1]](#1). Data structures and modules of the [VTK](https://vtk.org/) library is heavily relied upon. 

The implementation captured here is intended to form a basis for study of bin packing problems, with applications in scheduling and production planning for 3D printing. 

__The Orthogonal Minimum Container Packing problem__: 

Given a collection of objects, find the translation and rotation which has to be applied to each object, such that the volume of the axis-aligned bounding box of all the objects is minimised. 

We take the sum of the volumes of the axis-aligned bounding boxes of all the objects, individually, as a reference volume by which the objective is normalised. This is of course the smallest volume container that can possibily be achieved if all the objects are axis-aligned cuboids (of particular sizes), or modelled as such.

## Setup

On Linux

`python3 -m venv ./env`
`source env/bin/activate`
`pip install -r requirements.txt`

## References
<a id="1">[1]</a>
Alt, H., & Scharf, N. (2018). 
Approximating smallest containers for packing three-dimensional convex objects. International Journal of Computational Geometry & Applications, 28(02), 111-128.
