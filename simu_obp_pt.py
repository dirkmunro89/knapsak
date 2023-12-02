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
def back_da_pt(xk,fk,context,args):
#
    [n,pnts,maps,exts,c_l,c_r,c_a,c_v,nums,vtps,vtcs,int_flg,str_flg,log,vis,out]=args
#
#   [f,c]=simu_obp_co(xk,n,cols,tfms,vtps,exts,maps,c_l,c_r,c_a,c_v,int_flg,1)
#
    k=1
    for file in os.listdir(out):
        filename = os.fsdecode(file)
        if 'cubis_' in filename and filename.endswith(".vtp"):
            k=k+1
#
    log.info('%6d %14.3e'%(k,fk))
#
    app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,int_flg,str_flg,1)
    woutfle(out,app.GetOutput(),'cubis',k)
    app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,int_flg,str_flg,1)
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
def simu_obp_pt(xk,n,pnts,maps,exts,c_l,c_r,c_a,c_v,int_flg,flg):
#
    bnds=[]
    cens=[]
    encs=[]
    pnts_1=[]
#
#   apply transforms to point clouds to get new points
#       get new bounds etc. etc.
#
    for i in range(n):
#
        pnt=pnts[maps[i]]
#
        tmp = np.array([xk[7*i+1],xk[7*i+2],xk[7*i+3]])
        tmp = tmp/np.linalg.norm(tmp)
        rot=R.from_rotvec((c_a*xk[7*i])*tmp).as_matrix().T
        pnt_1=np.dot(pnt,rot)+xk[7*i+4:7*i+7]*c_l
#
        pnts_1.append(pnt_1)
        cens.append(c_l*xk[7*i+4:7*i+7])
        bnds.append(tuple(np.array([np.amin(pnt_1,axis=0).T,np.amax(pnt_1,axis=0).T]).T.flatten()))
#
    tax=np.amax(np.array(bnds),axis=0)
    tin=np.amin(np.array(bnds),axis=0)
    bds=[tin[0],tax[1],tin[2],tax[3],tin[4],tax[5]]
#
#   for each possible collision
#
    ct=0
    k=0
    for i in range(n-1):
        for j in range(i+1,n):
            if np.linalg.norm(cens[i]-cens[j])<1.1*(exts[maps[i]]+exts[maps[j]]):
                for p in pnts_1[i]:
                    tmp=np.amin(np.linalg.norm(p-pnts_1[j],axis=1))
                    if tmp < 100.:
                        ct=ct+1
                        break
            if ct>0:
                break
            k=k+1
        if ct>0:
            break
#   
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v+ct*1e1
#
    return f
#
