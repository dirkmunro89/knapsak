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
    [n,cols,tfms,vtps,exts,maps,c_l,c_r,c_a,c_v,nums,vtps,vtcs,int_flg,str_flg,log,vis,out]=args
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
def simu_obp_co(xk,n,cols,tfms,vtps,exts,maps,c_l,c_r,c_a,c_v,int_flg,flg):
#
    bnds=[]
    cens=[]
    encs=[]
    vtps_1=[]
#
#   apply transforms and compute new total bounding box
#
    for i in range(n):
#
        if int_flg == 24:
            cens.append(c_l*xk[4*i+1:4*i+4])
        else:
            cens.append(c_l*xk[7*i+4:7*i+7])
#
        tfmx(xk,i,c_l,c_a,c_r,tfms[i],int_flg,0)
#
        vtp=tran(vtps[maps[i]],tfms[i]) # can maybe get this from col object
        bnds.append(np.array(vtp.GetBounds()))
#
        vtps_1.append(vtp)
#       enc=vtk.vtkSelectEnclosedPoints()
#       enc.CheckSurfaceOff()
#       enc.Initialize(vtp)
#       encs.append(enc)
#          
    tax=np.amax(np.array(bnds),axis=0)
    tin=np.amin(np.array(bnds),axis=0)
    bds=[tin[0],tax[1],tin[2],tax[3],tin[4],tax[5]]
#
#   get collisions
#
    ct=0
    k=0
    enc_lst=[]
    enc_prs=[]
    for i in range(n-1):
        for j in range(i+1,n):
            if np.linalg.norm(cens[i]-cens[j])<1.1*(exts[maps[i]]+exts[maps[j]]):
                cols[k].Update()
                c=cols[k].GetNumberOfContacts()
                if c==0:
                    enc_lst.append(i)
                    enc_lst.append(j)
                    enc_prs.append((i,j))
#                   ct=ct+encs[i].IsInsideSurface(cens[j])+encs[j].IsInsideSurface(cens[i])
                ct=ct+c
                if ct > 0:
                    break
            if ct > 0:
                break
            k=k+1
        if ct > 0:
            break
#
    if ct==0:
        encs={}
        if len(enc_lst)>0:
            enc_lst=list(set(enc_lst))
            for e in enc_lst:
                enc=vtk.vtkSelectEnclosedPoints()
                enc.CheckSurfaceOff()
                enc.Initialize(vtps_1[e])
                encs[e]=enc
            for p in enc_prs:
                c1=encs[p[0]].IsInsideSurface(cens[p[1]])
                if c1:
                    ct=c1
                    break
                c2=encs[p[1]].IsInsideSurface(cens[p[0]])
                if c2:
                    ct=c2
                    break
#   
#   revert
#
    for i in range(n):
#
        tfmx(xk,i,c_l,c_a,c_r,tfms[i],int_flg,1)
#
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v+ct*n
#
    return f
#
