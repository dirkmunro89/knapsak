#
import os
import vtk
import numpy as np
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
#
from util import appdata, woutfle
#
def back_da(xk,fk,context,args):
#
    [n,pnts,maps,c_l,c_a,c_r,c_v,nums,vtps,vtcs,int_flg,str_flg,log,vis]=args
#
    [f,c]=simu_obp(xk,n,pnts,maps,c_l,c_a,c_r,c_v,int_flg,1)
    log.info('%14.3e %6d'%(fk,c))
#
    k=1
    for file in os.listdir('./'):
        filename = os.fsdecode(file)
        if 'cubis_' in filename and filename.endswith(".vtp"):
            k=k+1
#
    app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,int_flg,str_flg,1)
    woutfle(app.GetOutput(),'cubis',k)
    app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,int_flg,str_flg,1)
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
def simu_obp(xk,n,pnts,maps,c_l,c_a,c_r,c_v,int_flg,flg):
#
    bnds=[]
    vtps=[]
    npts=[]
#
#   apply transforms and compute new total bounding box
#
    c_v_n=0
#
    for i in range(n):
#
        pts_i=pnts[maps[i]]
#
        if int_flg == 1:
#
            tmp=xk[i*4]*3.#abs(xk[i*4])%7 - 3.5
#
            if tmp >= 0-3.5 and tmp < 1-3.5:
                rot=c_r[0]
            elif tmp >= 1-3.5 and tmp < 2-3.5:
                rot=c_r[1]
            elif tmp >= 2-3.5 and tmp < 3-3.5:
                rot=c_r[2]
            elif tmp >= 3-3.5 and tmp < 4-3.5:
                rot=c_r[3]
            elif tmp >= 4-3.5 and tmp < 5-3.5:
                rot=c_r[4]
            elif tmp >= 5-3.5 and tmp < 6-3.5:
                rot=c_r[5]
            elif tmp >= 6-3.5 and tmp < 7-3.5:
                rot=c_r[6]
            else:
                print(xk[i*4],tmp)
                print('error simu')
                exit()
            npts_i = np.dot(pts_i,rot) + xk[4*i+1:4*i+4]*c_l
#
        elif int_flg == 2:
#
            tmp=xk[i*7]*3.#abs(xk[i*4])%7 - 3.5
#
            if tmp >= 0-3.5 and tmp < 1-3.5:
                rot=c_r[0]
            elif tmp >= 1-3.5 and tmp < 2-3.5:
                rot=c_r[1]
            elif tmp >= 2-3.5 and tmp < 3-3.5:
                rot=c_r[2]
            elif tmp >= 3-3.5 and tmp < 4-3.5:
                rot=c_r[3]
            elif tmp >= 4-3.5 and tmp < 5-3.5:
                rot=c_r[4]
            elif tmp >= 5-3.5 and tmp < 6-3.5:
                rot=c_r[5]
            elif tmp >= 6-3.5 and tmp < 7-3.5:
                rot=c_r[6]
            else:
                print(xk[i*4],tmp)
                print('error simu')
                exit()
#
            npts_i = np.dot(pts_i,rot)*np.array([2.+xk[7*i+1:7*i+4]])+xk[7*i+4:7*i+7]*c_l
#
        else:
#
            tmp = np.array([xk[7*i+1],xk[7*i+2],xk[7*i+3]])
            tmp = tmp/np.linalg.norm(tmp)
#           tmp = tmp*c_a*xk[7*i]
            rot=R.from_rotvec( np.deg2rad(c_a*xk[7*i])*tmp ).as_matrix().T
            npts_i=np.dot(pts_i,rot)+xk[7*i+4:7*i+7]*c_l
#
        npts.append(npts_i)
#
        bnds.append(tuple(np.array([np.amin(npts_i,axis=0).T,np.amax(npts_i,axis=0).T]).T.flatten()))
#
        c_v_n=c_v_n+(bnds[i][1]-bnds[i][0])*(bnds[i][3]-bnds[i][2])*(bnds[i][5]-bnds[i][4])
#
        if i == 0:
            bds=list(bnds[i][:])
        else:
            for j in range(6):
                if j%2 == 0:
                    bds[j]=min(bds[j],bnds[i][j])
                else:
                    bds[j]=max(bds[j],bnds[i][j])
#
#   get collisions
#
    c=0
    m = int(n*(n-1)/2)
    for i in range(n-1):
        for j in range(i+1,n):
#
            pts_i=npts[i]
            com_i=np.sum(pts_i,axis=0)/8
            pts_j=npts[j]
            com_j=np.sum(pts_j,axis=0)/8
            bnd_i=bnds[i]
            bnd_j=bnds[j]
            ext_i=np.array([bnd_i[1]-bnd_i[0],bnd_i[3]-bnd_i[2],bnd_i[5]-bnd_i[4]])/2.
            ext_j=np.array([bnd_j[1]-bnd_j[0],bnd_j[3]-bnd_j[2],bnd_j[5]-bnd_j[4]])/2.
            if ( abs(com_i[0] - com_j[0]) <  ext_i[0] + ext_j[0] ):
                if ( abs(com_i[1] - com_j[1]) <  ext_i[1] + ext_j[1] ):
                    if ( abs(com_i[2] - com_j[2]) <  ext_i[2] + ext_j[2] ):
                        c=c+1
#
    if int_flg == 2:
        f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v_n
    else:
        f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    f=f+c
#
    if flg == 0:
        return f
    else:
        return f,c
#
