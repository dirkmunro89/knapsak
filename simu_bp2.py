#
import vtk
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
import numpy as np
from util import tran, appd3, woutfle
#
#   the default is premultiply; hence it looks strange that translation is before rotation
#
def back_bp3(xk,fk,context,args):
#
    [n,pnts,maps,c_l,c_r,c_v,nums,stps,stcs]=args
#
    [f,c]=simu_bp(xk,n,pnts,maps,c_l,c_r,c_v,1)
    print('%14.3e %6d %6d'%(fk,c,context),flush=True)
#
    app=appd3(xk,n,nums,maps,stcs,c_l,c_r)
    woutfle(app.GetOutput(),'cubes',0)
    app=appd3(xk,n,nums,maps,stps,c_l,c_r)
    woutfle(app.GetOutput(),'parts',0)
#
    if context==2:
        return True
    return False
#
def back_bp2(xk,args):
#
    n=args[0]
    nobj=args[1]
    cubs_str=args[2]
    objs_num=args[3]
    c_l=args[4]
    c_a=args[5]
    c_v=args[6]
    cols=args[7]
    tfms=args[8]
    maps=args[9]
    objs_str=args[10]
    objs=args[11]
    pnts=args[12]
    [f,c,outs]=simu_bp(xk,n,cols,tfms,objs,pnts,maps,c_l,c_a,c_v,1)
    print('%14.3e %6d'%(f,c),flush=True)
#
    app=appd3(xk,nobj,cubs_str,objs_num,c_l,c_a)
    woutfle(app.GetOutput(),'see',-1)
#
    return False
#
def back_bp(xk,convergence,args):
#
    n=args[0]
    nobj=args[1]
    cubs_str=args[2]
    objs_num=args[3]
    c_l=args[4]
    c_a=args[5]
    c_v=args[6]
    cols=args[7]
    tfms=args[8]
    maps=args[9]
    objs_str=args[10]
    objs=args[11]
    pnts=args[12]
    [f,c,outs]=simu_bp(xk,n,pnts,maps,c_r,c_v,1)
    print('%7.3f %14.3e %6d'%(convergence,f,c),flush=True)
#
    app=appd3(xk,nobj,cubs_str,objs_num,c_l,c_a)
    woutfle(app.GetOutput(),'see',-1)
#
    return False
#
def simu_bp(xk,n,pnts,maps,c_l,c_r,c_v,flg):
#def simu_bp(x,n,cols,tfms,objs,pnts,maps,c_l,c_a,c_v,flg):
#
    bnds=[]
    vtps=[]
    npts=[]
    for i in range(n):
#
        pts_i=pnts[maps[i]]
#
        if xk[i*4+3] >= 0-3.5 and xk[i*4+3] < 1-3.5:
            rot=c_r[0]
        elif xk[i*4+3] >= 1-3.5 and xk[i*4+3] < 2-3.5:
            rot=c_r[1]
        elif xk[i*4+3] >= 2-3.5 and xk[i*4+3] < 3-3.5:
            rot=c_r[2]
        elif xk[i*4+3] >= 3-3.5 and xk[i*4+3] < 4-3.5:
            rot=c_r[3]
        elif xk[i*4+3] >= 4-3.5 and xk[i*4+3] < 5-3.5:
            rot=c_r[4]
        elif xk[i*4+3] >= 5-3.5 and xk[i*4+3] < 6-3.5:
            rot=c_r[5]
        elif xk[i*4+3] >= 6-3.5 and xk[i*4+3] < 7-3.5:
            rot=c_r[6]
        else:
            print(xk[i*4+3])
            print('error')
            exit()
#
        npts_i = np.dot(pts_i,rot) + xk[4*i:4*i+3]*c_l
#
        npts.append(npts_i)
#
        bnds.append(tuple(np.array([np.amin(npts_i,axis=0).T,np.amax(npts_i,axis=0).T]).T.flatten()))
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
#   ext=100.
#   f=bds[5]+ext
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    f=f+c #est. volume per triangle
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
