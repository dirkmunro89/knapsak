from tvtk.api import tvtk
from vtkmodules.vtkFiltersSources import vtkSphereSource
import vtk.util.pickle_support
import pickle

sphereSrc = vtk.vtkSphereSource()
sphereSrc.Update()

pickled = pickle.dumps(sphereSrc.GetOutput())

unpickled = pickle.loads(pickled)

print(unpickled)

