#
#   orthogonal box packing
#
import vtk
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
import numpy as np
from util import tran, appdata, woutfle
#
#   the default is premultiply; hence it looks strange that translation is before rotation
#
def back_da_co(xk,fk,context,args):
#
    [n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v,nums,vtps,vtcs,int_flg,str_flg]=args
#
    [f,c]=simu_obp_co(xk,n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v,int_flg,1)
    print('%14.3e %6d %6d'%(fk,c,context),flush=True)
#
    app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'cubes',0)
    app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'parts',0)
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
#           derivative of modulo operator is 1 (in fixed point arithmetic) :)
            tmp=abs(xk[i*4])%7 - 3.5
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
            tmp=abs(xk[i*4])%7 - 3.5
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
#   if flg == 1:
#       print('Bounds: ', bds)
#       print('BBV: ', (bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4]))
#       print('c_v: ', c_v)
#       print('f: ', f)
#
#   b=0.
#   y=0.
#   if bds[0]<-ext:
#       b=b+(abs(bds[0])-ext)**1.
#       y=y+1
#   if bds[1]>ext:
#       b=b+(abs(bds[1])-ext)**1.
#       y=y+1
#   if bds[2]<-ext:
#       b=b+(abs(bds[2])-ext)**1.
#       y=y+1
#   if bds[3]>ext:
#       b=b+(abs(bds[3])-ext)**1.
#       y=y+1
#   if bds[4]<-ext:
#       b=b+(abs(bds[4]))**1.
#       y=y+1
#
    if flg == 0:
        return f
    else:
        return f,c
#
