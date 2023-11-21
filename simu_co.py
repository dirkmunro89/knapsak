#
import vtk
import numpy as np
from util import tran,appd,woutfle
#
def back_co(xk,fk,context,args):
#
    [n,cols,tfms,vtps,maps,c_l,c_a,c_v,nums,stps,stcs]=args
    [f,c]=simu_co(xk,n,cols,tfms,vtps,maps,c_l,c_a,c_v,1)
#   print('%14.3e %6d'%(f,c),flush=True)
    print('%14.3e %6d %6d'%(fk,c,context),flush=True)
#
    app=appd(xk,n,nums,maps,stps,c_l,c_a)
    woutfle(app.GetOutput(),'parts',1)
    app=appd(xk,n,nums,maps,stcs,c_l,c_a)
    woutfle(app.GetOutput(),'cubes',1)
#
    return False
#
def simu_co(x,n,cols,tfms,objs,maps,c_l,c_a,c_v,flg):
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
#   ext=100.
#   f=bds[5]+ext
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    if flg == 1:
        print('Bounds: ', bds)
        print('BBV: ', (bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4]))
        print('c_v: ', c_v)
        print('f: ', f)
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
