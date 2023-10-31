#
import os
import sys
import vtk
import pickle
import numpy as np
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
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
#   writer.SetCompressorTypeToZLib()
    writer.SetWriteToOutputString(True)
#   writer.SetFileName(fln+'_%d.vtp'%k)
    writer.Update()
    return writer.GetOutputString()
#
def simu(x,n,string):
#
    tfms=[]
    bnds=[]
    objs=[]
    for i in range(n):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(string[i])
        red.Update()
        obj = red.GetOutput()
        objs.append(obj)
#
        tfm = vtk.vtkTransform()
        if i < n-1:
            tfm.Translate(x[i*7+0], x[i*7+1], x[i*7+2])
            tfm.RotateWXYZ(x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
        else:
            tfm.RotateWXYZ(x[i*7+0], x[i*7+1], x[i*7+2], x[i*7+3])
        tfm.Update()
#
        vtp=tran(obj,tfm)
        bnds.append(vtp.GetBounds())
        tfms.append(tfm)
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
            c=c+coli.GetNumberOfContacts()
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
        obj=wout(obj,'ref',i)
#
        objs.append(obj)
#
    l=[-1e3]*(7*(n-1)+4)
    u=[1e3]*(7*(n-1)+4)
#
    res=differential_evolution(simu,args=(n,objs),callback=conv,bounds=list(zip(l,u)),workers=-1,seed=69,polish=True,maxiter=10000, init='sobol')
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
