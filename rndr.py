#
import vtk
#
def rndr(app):
#
    mpr = vtk.vtkPolyDataMapper()
    mpr.SetInputConnection(app.GetOutputPort())
    mpr.SetColorModeToDirectScalars()
#
    act = vtk.vtkActor()
    act.SetMapper(mpr)
    act.GetProperty().SetOpacity(1)
    act.GetProperty().SetBackfaceCulling(1)
#
    ren = vtk.vtkRenderer()
    ren.AddActor(act)
    ren.SetBackground(255,255,255)
    ren.UseHiddenLineRemovalOn()
#
    win = vtk.vtkRenderWindow()
    tmp=win.GetScreenSize()
    tmpx=int(tmp[0]/3)
    tmpy=int(tmp[0]/3)
    win.SetSize(tmpx, tmpy)
    win.SetPosition(0, 0)
    win.AddRenderer(ren)
#
    out=vtk.vtkOutlineFilter()
    out.SetInputConnection(app.GetOutputPort())
    box_mpr = vtk.vtkPolyDataMapper()
    box_mpr.SetInputConnection(out.GetOutputPort())
    box_act = vtk.vtkActor()
    box_act.SetMapper(box_mpr)
    box_act.GetProperty().SetColor(0,0,0)
    box_act.GetProperty().SetOpacity(1)
#
    ren.AddActor(box_act)
#
    ren.GetActiveCamera().Azimuth(45)
    ren.GetActiveCamera().Elevation(-90+35.264)
    ren.GetActiveCamera().SetViewUp(0,0,1)#35.264)
#
    ren.GetActiveCamera().ParallelProjectionOn()
#
#   ict=vtk.vtkRenderWindowInteractor()
#   ict.SetRenderWindow(win)
#
    win.SetWindowName('Stacking')
#
    bds=app.GetOutput().GetBounds()
#
    axs_act=vtk.vtkAxesActor()
    axs_act.SetTotalLength(0.25*(bds[1]-bds[0]), 0.25*(bds[3]-bds[2]), 0.25*(bds[5]-bds[4]))
    transform = vtk.vtkTransform()
    vtp=app.GetOutput()
    bds=vtp.GetBounds()
    transform.Translate(bds[0],bds[2],bds[4])
    axs_act.SetUserTransform(transform)
    axs_act.SetAxisLabels(0)
#
    act.GetProperty().SetInterpolationToFlat()

    ren.AddActor(axs_act)
#
    ren.ResetCameraScreenSpace()
    win.Render()
#
    vis=[mpr,box_mpr,axs_act,ren,win]
#
    return vis
#
