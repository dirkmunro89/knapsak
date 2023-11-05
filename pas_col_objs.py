#
import os
import sys
import vtk
import numpy as np
import multiprocessing as mp
from scipy.optimize import differential_evolution
#
def conv(xk,convergence):
#
    print(convergence)
    return False
#
def tran(vtp,tfm):
#
    tfm_flt =  vtk.vtkTransformPolyDataFilter()
    tfm_flt.SetInputData(vtp)
    tfm_flt.SetTransform(tfm)
    tfm_flt.Update()
    vtp = tfm_flt.GetOutput()
#
    return vtp
#
def move(vtp,trs,rot):
#
    tfm = vtk.vtkTransform()
    tfm.Translate(trs[0], trs[1], trs[2])
    tfm.RotateWXYZ(rot[0], rot[1], rot[2], rot[3])
    tfm.Update()
    tfm_flt =  vtk.vtkTransformPolyDataFilter()
    tfm_flt.SetInputData(vtp)
    tfm_flt.SetTransform(tfm)
    tfm_flt.Update()
    vtp = tfm_flt.GetOutput()
#
    return [vtp,tfm,tfm_flt]
#
def wout(vtp,fln,k):
#
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtp)
    writer.SetFileName(fln+'_%d.vtp'%k)
    writer.Update()
#
def simu(x,aux):
#
    n=aux[0]
    cols=aux[1]
    tfms=aux[2]
    objs=aux[3]
#
#   invert the current transform (understood to take it back to ref)
#
    bnds=[]
    m = int(n*(n-1)/2)
    for i in range(n):
        if i < n-1:
            tfms[i].Translate(x[i*7+0], x[i*7+1], x[i*7+2])
            tfms[i].RotateWXYZ(x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
        else:
            tfms[i].RotateWXYZ(x[i*7+0], x[i*7+1], x[i*7+2], x[i*7+3])
        vtp=tran(objs[i],tfms[i])
        bnds.append(vtp.GetBounds())
    c=0
    for i in range(m):
        cols[i].Update()
        c=c+cols[i].GetNumberOfContacts()
    for i in range(n):
        if i < n-1:
            tfms[i].RotateWXYZ(-x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
            tfms[i].Translate(-x[i*7+0], -x[i*7+1], -x[i*7+2])
        else:
            tfms[i].RotateWXYZ(-x[i*7+0], x[i*7+1], x[i*7+2], x[i*7+3])
#
    for j in range(n):
        if j == 0:
            bds=list(bnds[j][:])
        else:
            for i in range(6):
                if i%2 == 0:
                    bds[i] = min(bds[i],bnds[j][i])
                else:
                    bds[i] = max(bds[i],bnds[j][i])
#
    f=c*1e6
    f=f+(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])# + np.linalg.norm(x[:3])**2.
#
    return f
#
if __name__ == "__main__":
#
    n=int(sys.argv[1])
    fln=sys.argv[2]
#
#   read in parts
#
    objs=[]
    tfms=[]
#
    for i in range(n):
#
        red = vtk.vtkSTLReader()
        red.SetFileName(fln)
        red.Update()
        obj = red.GetOutput()
#
        com = vtk.vtkCenterOfMass()
        com.SetInputData(obj)
        com.SetUseScalarsAsWeights(False)
        com.Update()
        g = np.array(com.GetCenter())
#
        [obj,tfm_0,_] = move(obj,-g,[0.,1.,0.,0.])
#
        [_,tfm,_] = move(obj,[0.,0.,0.],[0.,1.,0.,0.])
#
        objs.append(obj)
        tfms.append(tfm)
#
    cols=[]
#
    for i in range(n):
        for j in range(i+1,n):
#
            coli = vtk.vtkCollisionDetectionFilter()
            coli.SetCollisionModeToFirstContact()
            coli.SetInputData(0, objs[i])
            coli.SetTransform(0, tfms[i])
            coli.SetInputData(1, objs[j])
            coli.SetTransform(1, tfms[j])
            coli.Update()
#
            cols.append(coli)
#
    x0=np.zeros(7*n)
#
    l=[-1e3]*(7*(n-1)+4)
    u=[1e3]*(7*(n-1)+4)
#
    aux=[n,cols,tfms,objs]
#
    res=differential_evolution(simu,args=(aux,),callback=conv,bounds=list(zip(l,u)),workers=1,seed=69,polish=True,maxiter=1000, init='sobol')
    print(res)
    x=res.x
    for i in range(n):
        if i < n-1:
            tmp = x[7*i:7*i+7]
        else:
            tmp=np.zeros(7)
            tmp[3:]=x[7*i:7*i+4]
        [vtp,_,_] = move(objs[i],tmp[:3],tmp[3:7])
        wout(vtp,'see',i)
#
