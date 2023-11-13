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
    print('%14.3e %6d %6d'%(fk,c, context),flush=True)
#
    app=appd3(xk,nobj,cubs_str,objs_num,c_l,c_a)
    woutfle(app.GetOutput(),'see',-1)
    app=appd3(xk,nobj,objs_str,objs_num,c_l,c_a)
    woutfle(app.GetOutput(),'lee',-1)
#
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
    [f,c,outs]=simu_bp(xk,n,cols,tfms,objs,pnts,maps,c_l,c_a,c_v,1)
    print('%7.3f %14.3e %6d'%(convergence,f,c),flush=True)
#
    app=appd3(xk,nobj,cubs_str,objs_num,c_l,c_a)
    woutfle(app.GetOutput(),'see',-1)
#
    return False
#
def simu_bp(x,n,cols,tfms,objs,pnts,maps,c_l,c_a,c_v,flg):
#
    bnds=[]
    vtps=[]
    new_pts=[]
#   pnts=[]
    for i in range(n):
#
#       tfms[i].Translate(c_l[0]*x[i*4+0], c_l[1]*x[i*4+1], c_l[2]*x[i*4+2])
        pts_i=pnts[maps[i]]#numpy_support.vtk_to_numpy(objs[maps[i]].GetPoints().GetData())
#       print(pts_i)
#       pts_i=numpy_support.vtk_to_numpy(objs[maps[i]].GetPoints().GetData())
#       stop
#
#       operate on array of points from outside
#
        if x[i*4+3] >= 0-3.5 and x[i*4+3] < 1-3.5:
            new_pts_i=pts_i.copy()
        elif x[i*4+3] >= 1-3.5 and x[i*4+3] < 2-3.5:
            rot=R.from_rotvec(2*np.pi/3 * np.array([1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)])).as_matrix().T
            new_pts_i = np.dot(pts_i,rot)
        elif x[i*4+3] >= 2-3.5 and x[i*4+3] < 3-3.5:
            rot=R.from_rotvec(np.pi/2 * np.array([1,0,0])).as_matrix().T
            new_pts_i = np.dot(pts_i,rot)
        elif x[i*4+3] >= 3-3.5 and x[i*4+3] < 4-3.5:
            rot=R.from_rotvec(np.pi/2 * np.array([0,1,0])).as_matrix().T
            new_pts_i = np.dot(pts_i,rot)
        elif x[i*4+3] >= 4-3.5 and x[i*4+3] < 5-3.5:
            rot=R.from_rotvec(np.pi/2 * np.array([0,0,1])).as_matrix().T
            new_pts_i = np.dot(pts_i,rot)
        elif x[i*4+3] >= 5-3.5 and x[i*4+3] < 6-3.5:
            rot=R.from_rotvec(2*np.pi/3 * np.array([-1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)])).as_matrix().T
            new_pts_i = np.dot(pts_i,rot)
        elif x[i*4+3] >= 6-3.5 and x[i*4+3] < 7-3.5:
            new_pts_i=pts_i.copy()
        else:
            print(x[i*4+3])
            print('error')
            exit()
#
        new_pts_i = new_pts_i + x[4*i:4*i+3]*c_l
#
        new_pts.append(new_pts_i)
#       vtp=tran(objs[maps[i]],tfms[i])
#       bnds.append(vtp.GetBounds())
#       vtps.append(vtp)
#
        bnds.append(  tuple( np.array([np.amin(new_pts_i,axis=0).T,  np.amax(new_pts_i,axis=0).T]).T.flatten()  ) )
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
            pts_i=new_pts[i]#numpy_support.vtk_to_numpy(vtps[i].GetPoints().GetData())
#           pts_i=numpy_support.vtk_to_numpy(vtps[i].GetPoints().GetData())
            com_i=np.sum(pts_i,axis=0)/8
            pts_j=new_pts[j]#numpy_support.vtk_to_numpy(vtps[j].GetPoints().GetData())
#           pts_j=numpy_support.vtk_to_numpy(vtps[j].GetPoints().GetData())
            com_j=np.sum(pts_j,axis=0)/8
            bnd_i=bnds[i]
            bnd_j=bnds[j]
            sze_i=np.array([bnd_i[1]-bnd_i[0],bnd_i[3]-bnd_i[2],bnd_i[5]-bnd_i[4]])/2.
            sze_j=np.array([bnd_j[1]-bnd_j[0],bnd_j[3]-bnd_j[2],bnd_j[5]-bnd_j[4]])/2.
            if ( abs(com_i[0] - com_j[0]) <  sze_i[0] + sze_j[0] ):
                if ( abs(com_i[1] - com_j[1]) <  sze_i[1] + sze_j[1] ):
                    if ( abs(com_i[2] - com_j[2]) <  sze_i[2] + sze_j[2] ):
                        c=c+1
#
#   for i in range(m):
#       cols[i].Update()
#       c=c+cols[i].GetNumberOfContacts()
#
#   revert
#
#   for i in range(n):
#       if x[i*4+3] >= 0 and x[i*4+3] < 1:
#           tfms[i].RotateWXYZ(0, 1, 0, 0)
#       elif x[i*4+3] >= 1 and x[i*4+3] < 2:
#           tfms[i].RotateWXYZ(-90, 1, 0, 0)
#       elif x[i*4+3] >= 2 and x[i*4+3] < 3:
#           tfms[i].RotateWXYZ(-90, 0, 1, 0)
#       elif x[i*4+3] >= 3 and x[i*4+3] < 4:
#           tfms[i].RotateWXYZ(-90, 0, 0, 1)
#       elif x[i*4+3] >= 4 and x[i*4+3] < 5:
#           tfms[i].RotateWXYZ(-120, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3))
#       elif x[i*4+3] >= 5 and x[i*4+3] < 6:
#           tfms[i].RotateWXYZ(-120, -1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3))
#       elif x[i*4+3] >= 6 and x[i*4+3] < 7:
#           tfms[i].RotateWXYZ(0, 1, 0, 0)
#       else:
#           print('error')
#           exit()
#       tfms[i].Translate(-c_l[0]*x[i*4+0], -c_l[1]*x[i*4+1], -c_l[2]*x[i*4+2])
#
    ext=100.
    f=bds[5]+ext
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    f=f+c #est. volume per triangle
#
    b=0.
    y=0.
    if bds[0]<-ext:
        b=b+(abs(bds[0])-ext)**1.
        y=y+1
    if bds[1]>ext:
        b=b+(abs(bds[1])-ext)**1.
        y=y+1
    if bds[2]<-ext:
        b=b+(abs(bds[2])-ext)**1.
        y=y+1
    if bds[3]>ext:
        b=b+(abs(bds[3])-ext)**1.
        y=y+1
    if bds[4]<-ext:
        b=b+(abs(bds[4]))**1.
        y=y+1
#
    if flg == 0:
        return f
    else:
        return f,c,vtps
#
