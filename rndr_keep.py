#
import vtk
#
def rndr(app):
#
    app_mpr=vtk.vtkPolyDataMapper()
    app_mpr.SetInputConnection(app.GetOutputPort())
    app_mpr.SetColorModeToDirectScalars()
#
    app_act=vtk.vtkActor()
    app_act.SetMapper(app_mpr)
#   app_act.GetProperty().SetOpacity(1.99)
#
    ren=vtk.vtkRenderer()
    ren.AddActor(app_act)
    ren.SetBackground(255,255,255)
    ren.UseHiddenLineRemovalOn()
#
    win=vtk.vtkRenderWindow()
    tmp=win.GetScreenSize()
    tmpx=int(tmp[0]/3)
    tmpy=int(tmp[0]/3)
    win.SetSize(tmpx, tmpy)
    win.SetPosition(0, 0)
    win.AddRenderer(ren)
#
    box=vtk.vtkOutlineFilter()
    box.SetInputConnection(app.GetOutputPort())
    box_mpr = vtk.vtkPolyDataMapper()
    box_mpr.SetInputConnection(box.GetOutputPort())
    box_act = vtk.vtkActor()
    box_act.SetMapper(box_mpr)
    box_act.GetProperty().SetColor(0,0,0)
    box_act.GetProperty().SetOpacity(0.75)
#
#
    ren.AddActor(box_act)
#
    ren.GetActiveCamera().Azimuth(45)
    ren.GetActiveCamera().Elevation(-90+35.264)
    ren.GetActiveCamera().SetViewUp(0,0,1)#35.264)
#
    ren.GetActiveCamera().ParallelProjectionOn()
#
    itr=vtk.vtkRenderWindowInteractor()
    itr.SetRenderWindow(win)
#
    win.SetWindowName('Stacking')
#
    bds=app.GetOutput().GetBounds()
#
    axs_act=vtk.vtkAxesActor()
    axs_act.SetTotalLength(0.25*(bds[1]-bds[0]), 0.25*(bds[3]-bds[2]), 0.25*(bds[5]-bds[4]))
    tfm=vtk.vtkTransform()
    vtp=app.GetOutput()
    bds=vtp.GetBounds()
    tfm.Translate(bds[0],bds[2],bds[4])
    axs_act.SetUserTransform(tfm)
    axs_act.SetAxisLabels(0)
#
    app_act.GetProperty().SetInterpolationToFlat()

    ren.AddActor(axs_act)
#
    ren.ResetCameraScreenSpace()
    ren.Render()
#
    vis=[app_mpr,box_mpr,axs_act,ren,win]
#
    return vis
#
