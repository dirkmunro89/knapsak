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
    [n,cols,tfms,vtps,exts,maps,c_l,c_r,c_a,c_s,c_v,nums,vtps,vtcs,int_flg,str_flg,log,vis,out]=args
#
    [f,c]=simu_obp_co(xk,n,cols,tfms,vtps,exts,maps,c_l,c_r,c_a,c_s,c_v,int_flg,1)
    log.info('%14.3e %6d'%(fk,c))
#
    k=1
    for file in os.listdir(out):
        filename = os.fsdecode(file)
        if 'cubis_' in filename and filename.endswith(".vtp"):
            k=k+1
#
    app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,np.ones(3),int_flg,str_flg,1)
    woutfle(out,app.GetOutput(),'cubis',k)
    app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,np.ones(3),int_flg,str_flg,1)
    woutfle(out,app.GetOutput(),'objec',k)
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
def simu_obp_co(xk,n,cols,tfms,vtps,exts,maps,c_l,c_r,c_a,c_s,c_v,int_flg,flg):
#
    bnds=[]
    cens=[]
    encs=[]
#
#   apply transforms and compute new total bounding box
#
    for i in range(n):
#
        tfmx(xk,i,c_l,c_a,c_r,c_s,tfms[i],int_flg,0)
        vtp=tran(vtps[maps[i]],tfms[i]) # can maybe get this from col object
#
        flt=vtk.vtkCenterOfMass()
        flt.SetInputData(vtp)
        flt.SetUseScalarsAsWeights(False)
        flt.Update()
#
        cens.append(np.array(flt.GetCenter()))
        bnds.append(vtp.GetBounds())
#
        enc=vtk.vtkSelectEnclosedPoints()
        enc.CheckSurfaceOff()
        enc.Initialize(vtp)
        encs.append(enc)
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
    k=0
    for i in range(n-1):
        for j in range(i+1,n):
            if np.linalg.norm(cens[i]-cens[j]) < np.amax(c_s)*(exts[maps[i]]+exts[maps[j]]):
                tmp_c1=encs[i].IsInsideSurface(cens[j])
                tmp_c1=tmp_c1+encs[j].IsInsideSurface(cens[i])
                if tmp_c1==0:
                    cols[k].Update()
                    c=c+cols[k].GetNumberOfContacts()
                c=c+tmp_c1
            k=k+1
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
