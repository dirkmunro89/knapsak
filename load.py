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
    ren = vtk.vtkRenderer()
    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetWindowName("Packing")
    interact = vtk.vtkRenderWindowInteractor()
    interact.SetRenderWindow(win)
#
    tmp=win.GetScreenSize()
    tmpx=int(tmp[0]/3)
    tmpy=int(tmp[0]/3)
    win.SetSize(tmpx, tmpy)
    win.SetPosition(0, 0)
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
    ren.AddActor(actor)
    ren.AddActor(outline_actor)
    ren.AddActor(axes_actor)
    ren.SetBackground(255, 255, 255) 
#
    ren.ResetCameraScreenSpace(bds)
    win.Render()
    ren.UseHiddenLineRemovalOn()
    interact.Start()
#
