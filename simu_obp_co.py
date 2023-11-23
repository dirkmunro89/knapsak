#
import os
import vtk
import numpy as np
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
#
from util import tran, appdata, woutfle
#
#   the default is premultiply; hence it looks strange that translation is before rotation
#
def back_da_co(xk,fk,context,args):
#
    [n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v,nums,vtps,vtcs,int_flg,str_flg,log]=args
#
    [f,c]=simu_obp_co(xk,n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v,int_flg,1)
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
    if context==2:
        return True
    return False
#
def simu_obp_co(xk,n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v,int_flg,flg):
#
    bnds=[]
#
#   apply transforms and compute new total bounding box
#
    for i in range(n):
#
        if int_flg:
#   
#           derivative of modulo operator is 1 :)
            tmp=xk[i*4]#abs(xk[i*4])%7 - 3.5
#
            if tmp >= 0-3.5 and tmp < 1-3.5:
                r=R.from_matrix(c_r[0].T).as_rotvec()
            elif tmp >= 1-3.5 and tmp < 2-3.5:
                r=R.from_matrix(c_r[1].T).as_rotvec()
            elif tmp >= 2-3.5 and tmp < 3-3.5:
                r=R.from_matrix(c_r[2].T).as_rotvec()
            elif tmp >= 3-3.5 and tmp < 4-3.5:
                r=R.from_matrix(c_r[3].T).as_rotvec()
            elif tmp >= 4-3.5 and tmp < 5-3.5:
                r=R.from_matrix(c_r[4].T).as_rotvec()
            elif tmp >= 5-3.5 and tmp < 6-3.5:
                r=R.from_matrix(c_r[5].T).as_rotvec()
            elif tmp >= 6-3.5 and tmp < 7-3.5:
                r=R.from_matrix(c_r[6].T).as_rotvec()
            else:
                print(xk[i*4],tmp)
                print('error util')
                exit()
            tmp=max(np.linalg.norm(r),1e-9)
            tfms[i].RotateWXYZ(np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
            tfms[i].Translate(c_l[0]*xk[i*4+1], c_l[1]*xk[i*4+2], c_l[2]*xk[i*4+3])
#
        else:
            tfms[i].RotateWXYZ(c_a*xk[i*7], xk[i*7+1], xk[i*7+2], xk[i*7+3])
            tfms[i].Translate(c_l[0]*xk[i*7+4], c_l[1]*xk[i*7+5], c_l[2]*xk[i*7+6])
#
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
        if int_flg:
#   
#           derivative of modulo operator is 1 (in fixed point arithmetic) :)
            tmp=xk[i*4]#abs(xk[i*4])%7 - 3.5
#
            if tmp >= 0-3.5 and tmp < 1-3.5:
                r=R.from_matrix(c_r[0].T).as_rotvec()
            elif tmp >= 1-3.5 and tmp < 2-3.5:
                r=R.from_matrix(c_r[1].T).as_rotvec()
            elif tmp >= 2-3.5 and tmp < 3-3.5:
                r=R.from_matrix(c_r[2].T).as_rotvec()
            elif tmp >= 3-3.5 and tmp < 4-3.5:
                r=R.from_matrix(c_r[3].T).as_rotvec()
            elif tmp >= 4-3.5 and tmp < 5-3.5:
                r=R.from_matrix(c_r[4].T).as_rotvec()
            elif tmp >= 5-3.5 and tmp < 6-3.5:
                r=R.from_matrix(c_r[5].T).as_rotvec()
            elif tmp >= 6-3.5 and tmp < 7-3.5:
                r=R.from_matrix(c_r[6].T).as_rotvec()
            else:
                print(xk[i*4],tmp)
                print('error util')
                exit()
            tmp=max(np.linalg.norm(r),1e-9)
            tfms[i].Translate(-c_l[0]*xk[i*4+1], -c_l[1]*xk[i*4+2], -c_l[2]*xk[i*4+3])
            tfms[i].RotateWXYZ(-np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
        else:
            tfms[i].Translate(-c_l[0]*xk[i*7+4], -c_l[1]*xk[i*7+5], -c_l[2]*xk[i*7+6])
            tfms[i].RotateWXYZ(-c_a*xk[i*7], xk[i*7+1], xk[i*7+2], xk[i*7+3])
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
