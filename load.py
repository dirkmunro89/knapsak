#
import vtk
import sys
#
if __name__ == "__main__":
#
    fln=sys.argv[1]
#
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fln)
    reader.Update()
#
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    mapper.SetColorModeToDirectScalars()
#
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1.0)
#
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetWindowName("Packing")
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
#
    out=vtk.vtkOutlineFilter()
    out.SetInputConnection(reader.GetOutputPort())
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(out.GetOutputPort())
#
    outline_actor = vtk.vtkActor()
    outline_actor.SetMapper(mapper)
    outline_actor.GetProperty().SetColor(0,0,0)
    outline_actor.GetProperty().SetOpacity(0.75)
#
    axes_actor=vtk.vtkAxesActor()
    axes_actor.SetTotalLength(100, 100, 100)
    transform = vtk.vtkTransform()
#
    vtp=reader.GetOutput()
    bds=vtp.GetBounds()
#
    actor.GetProperty().SetInterpolationToFlat()
#
    transform.Translate(bds[0],bds[2],bds[4])
    axes_actor.SetUserTransform(transform)
    axes_actor.SetAxisLabels(0)
#
    renderer.AddActor(actor)
    renderer.AddActor(outline_actor)
    renderer.AddActor(axes_actor)
    renderer.SetBackground(255, 255, 255) 
#
    renderWindow.Render()
    renderWindowInteractor.Start()
#
