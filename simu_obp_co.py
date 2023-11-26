#
import os
import vtk
import numpy as np
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
#
from util import tfmx, tran, appdata, woutfle
#
#   the default is premultiply; hence it looks strange that translation is before rotation
#
def back_da_co(xk,fk,context,args):
#
    [n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_s,c_v,nums,vtps,vtcs,int_flg,str_flg,log,vis]=args
#
    [f,c]=simu_obp_co(xk,n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_s,c_v,int_flg,1)
    log.info('%14.3e %6d'%(fk,c))
#
    k=1
    for file in os.listdir('./'):
        filename = os.fsdecode(file)
        if 'cubis_' in filename and filename.endswith(".vtp"):
            k=k+1
#
    app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,np.ones(3),int_flg,str_flg,1)
    woutfle(app.GetOutput(),'cubis',k)
    app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,np.ones(3),int_flg,str_flg,1)
    woutfle(app.GetOutput(),'objec',k)
#
    if vis:
#
        [app_mpr,box_mpr,axs_act,ren,win]=vis
#
        app_mpr.SetInputConnection(app.GetOutputPort())
#
        box=vtk.vtkOutlineFilter()
        box.SetInputConnection(app.GetOutputPort())
        box_mpr.SetInputConnection(box.GetOutputPort())
#
        tfm = vtk.vtkTransform()
        vtp=app.GetOutput()
        bds=vtp.GetBounds()
        tfm.Translate(bds[0],bds[2],bds[4])
        axs_act.SetUserTransform(tfm)
#
        ren.ResetCameraScreenSpace(bds)
        win.Render()
#
    return False
#
def simu_obp_co(xk,n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_s,c_v,int_flg,flg):
#
    bnds=[]
#
#   apply transforms and compute new total bounding box
#
    for i in range(n):
#
        tfmx(xk,i,c_l,c_a,c_r,c_s,tfms[i],int_flg,0)
        vtp=tran(vtps[maps[i]],tfms[i])
        bnds.append(vtp.GetBounds())
#          
        if i == 0:
            bds=list(bnds[i][:])
        else:
            for j in range(6):
                if j%2 == 0:
                    bds[j] = min(bds[j],bnds[i][j])
                else:
                    bds[j] = max(bds[j],bnds[i][j])
#      
#   get collisions
#
    c=0
    m = int(n*(n-1)/2)
    for i in range(m):
        cols[i].Update()
        c=c+cols[i].GetNumberOfContacts()
#
#   revert
#
    for i in range(n):
#
        tfmx(xk,i,c_l,c_a,c_r,c_s,tfms[i],int_flg,1)
#
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    f=f+c
#
    if flg == 0:
        return f
    else:
        return f,c
#
