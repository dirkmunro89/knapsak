#
import vtk
import numpy as np
from util import tran
#
def back_nm_co(xk,args):
#
    n=args[0]
    objs=args[1]
    cols=args[2]
    tfms=args[3]
    c_l=args[4]
    c_a=args[5]
    c_v=args[6]
    [f,c]=simu_nm_co(xk,n,objs,maps,cols,tfms,c_l,c_a,c_v,1)
    print('%14.3e %6d'%(f,c),flush=True)
#
    return False
#
def simu_nm_co(x,n,cols,tfms,objs,maps,c_l,c_a,c_v,flg):
#
    bnds=[]
    for i in range(n):
#
        tfms[i].Translate(c_l[0]*x[i*7+0], c_l[1]*x[i*7+1], c_l[2]*x[i*7+2])
        tfms[i].RotateWXYZ(c_a*x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
        vtp=tran(objs[maps[i]],tfms[i])
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
    c=0
    m = int(n*(n-1)/2)
    for i in range(m):
#
        cols[i].Update()
        c=c+cols[i].GetNumberOfContacts()
#
#   revert
#
    for i in range(n):
#
        tfms[i].RotateWXYZ(-c_a*x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
        tfms[i].Translate(-c_l[0]*x[i*7+0], -c_l[1]*x[i*7+1], -c_l[2]*x[i*7+2])
#
#   c=0
#   m = int(n*(n-1)/2)
#   for i in range(m):
#
#       cols[i].Update()
#       c=c+cols[i].GetNumberOfContacts()
#   print(c)
#   stop
#
    ext=100.
    f=bds[5]+ext
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    f=f+c*c_v/n/309*1e6 #est. volume per triangle
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
        return f,c
#
