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
def back_x(xk,args):
#
    [n,pnts,maps,c_l,c_r,c_v,nums,vtps,vtcs,int_flg,str_flg]=args
#
    [f,c]=simu_obp(xk,n,pnts,maps,c_l,c_r,c_v,1)
    print('%14.3e %6d'%(f,c),flush=True)
#
    app=appdata(xk,n,nums,maps,vtcs,c_l,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'cubes',0)
    app=appdata(xk,n,nums,maps,vtps,c_l,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'parts',0)
#
    return False
#
def back_bh(xk,fk,accept,args):
#
    [n,pnts,maps,c_l,c_r,c_v,nums,vtps,vtcs,int_flg,str_flg]=args
#
    [f,c]=simu_obp(xk,n,pnts,maps,c_l,c_r,c_v,1)
    print('%14.3e %6d %6d'%(f,c,accept),flush=True)
#
    app=appdata(xk,n,nums,maps,vtcs,c_l,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'cubes',0)
    app=appdata(xk,n,nums,maps,vtps,c_l,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'parts',0)
#
    return False
#
def back_da(xk,fk,context,args):
#
    [n,pnts,maps,c_l,c_a,c_r,c_v,nums,vtps,vtcs,int_flg,str_flg]=args
#
    [f,c]=simu_obp(xk,n,pnts,maps,c_l,c_a,c_r,c_v,int_flg,1)
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
def back_de(xk,convergence,args):
#
    [n,pnts,maps,c_l,c_r,c_v,nums,stps,stcs,int_flg,str_flg]=args
#
    [f,c]=simu_obp(xk,n,pnts,maps,c_l,c_r,c_v,1)
    print('%14.3e %6d %14.3e'%(f,c,convergence),flush=True)
#
    app=appdata(xk,n,nums,maps,stcs,c_l,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'cubes',0)
    app=appdata(xk,n,nums,maps,stps,c_l,c_r,int_flg,str_flg)
    woutfle(app.GetOutput(),'parts',0)
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
def simu_obp(xk,n,pnts,maps,c_l,c_a,c_r,c_v,int_flg,flg):
#
    bnds=[]
    vtps=[]
    npts=[]
#
#   apply transforms and compute new total bounding box
#
    for i in range(n):
#
        pts_i=pnts[maps[i]]
#
#       derivative of modulo operator is 1 (in fixed point arithmetic) :)
#
        if int_flg:
            tmp=abs(xk[i*4])%7 - 3.5
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
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    f=f+c
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
