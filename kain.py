#
import os
import sys
import vtk
import pickle
import numpy as np
import multiprocessing
from functools import partial
from scipy.optimize import differential_evolution
#
#
def conv(xk,convergence,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu(xk,n,objs,c_l,c_a,c_v,1)
    print('%7.3f %14.3e %6d'%(convergence,f,c))
#
    for i in range(n):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
#
        if i < n-1:
            tmp = xk[7*i:7*i+7]
        else:
            tmp=np.zeros(7)
            tmp[3:]=xk[7*i:7*i+4]
        [vtp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        woutfle(vtp,'see',i)
#
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
def move(vtp,trs,rot,c_l,c_a):
#
    tfm = vtk.vtkTransform()
    tfm.Translate(c_l*trs[0], c_l*trs[1], c_l*trs[2])
    tfm.RotateWXYZ(c_a*rot[0], rot[1], rot[2], rot[3])
    tfm.Update()
    tfm_flt =  vtk.vtkTransformPolyDataFilter()
    tfm_flt.SetInputData(vtp)
    tfm_flt.SetTransform(tfm)
    tfm_flt.Update()
    vtp = tfm_flt.GetOutput()
#
    return [vtp,tfm,tfm_flt]
#
def woutfle(vtp,fln,k):
#
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtp)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    writer.SetFileName(fln+'_%d.vtp'%k)
    writer.Update()
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
def simu(x,n,string,c_l,c_a,c_v,flg):
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
            tfm.Translate(c_l*x[i*7+0], c_l*x[i*7+1], c_l*x[i*7+2])
            tfm.RotateWXYZ(c_a*x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
        else:
            tfm.RotateWXYZ(c_a*x[i*7+0], x[i*7+1], x[i*7+2], x[i*7+3])
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
            coli.SetCollisionModeToAllContacts()
            coli.SetCollisionModeToHalfContacts()
#           coli.SetCollisionModeToFirstContact()
            coli.SetInputData(0, objs[i])
            coli.SetTransform(0, tfms[i])
            coli.SetInputData(1, objs[j])
            coli.SetTransform(1, tfms[j])
            coli.Update()
            c=c+coli.GetNumberOfContacts()
#
    f=bds[5]-bds[4]# + np.linalg.norm(x[:3])**2.
#   f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])# + np.linalg.norm(x[:3])**2.
#
#   f=f/c_v 
    f=f+c*c_v/n/309 #est. volume per triangle
#
    b=0.
    ext=75.
    if bds[0]<-ext:
        b=b+abs(bds[0])-ext
    if bds[1]>ext:
        b=b+abs(bds[1])-ext
    if bds[2]<-ext:
        b=b+abs(bds[2])-ext
    if bds[3]>ext:
        b=b+abs(bds[3])-ext
#
    f=f+b*c_l
#
    if flg == 0:
        return f
    else:
        return f,c
#
if __name__ == "__main__":
#
    n=int(sys.argv[1])
    fln=sys.argv[2]
#
#
#   read in parts
#
    c_l=0.
    c_a=180.0
    objs=[]
    volt=0.
    c_v=0.
    for i in range(n):
#
        red = vtk.vtkSTLReader()
        red.SetFileName(fln)
        red.Update()
        obj = red.GetOutput()
#
        prp = vtk.vtkMassProperties()
        prp.SetInputData(obj)
        prp.Update() 
#
        print("Volume = ", prp.GetVolume())
        print("Surface = ", prp.GetSurfaceArea())
#
        volt=volt+prp.GetVolume()
#
        com = vtk.vtkCenterOfMass()
        com.SetInputData(obj)
        com.SetUseScalarsAsWeights(False)
        com.Update()
        g = np.array(com.GetCenter())
#
        bds=obj.GetBounds()
        c_l=max(bds[1]-bds[0],c_l)
        c_l=max(bds[3]-bds[2],c_l)
        c_l=max(bds[5]-bds[4],c_l)
        c_v = c_v+(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])
#
        [obj,tfm_0,_] = move(obj,-g,[0.,1.,0.,0.],1.,1.)
#
        obj=wout(obj,'ref',i)
#
        objs.append(obj)
#
    c_l=c_l*n*2
#
    l=[-1e0]*(7*(n-1)+4)
    u=[1e0]*(7*(n-1)+4)
#
    res=differential_evolution(simu,args=(n,objs,c_l,c_a,c_v,0),callback=partial(conv,args=(n,objs,c_l,c_a,c_v)),bounds=list(zip(l,u)),workers=2,polish=True,disp=True,maxiter=10000,updating='deferred')
    print(res)
    x=res.x
    for i in range(n):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
#
        if i < n-1:
            tmp = x[7*i:7*i+7]
        else:
            tmp=np.zeros(7)
            tmp[3:]=x[7*i:7*i+4]
        [vtp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        woutfle(vtp,'see',i)
#
