#
import os
import vtk
import numpy as np
from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint
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
def simu(x,coli,tfm_a,tfm_b,prt_a,prt_b):
#
#   invert the current transform (understood to take it back to ref)
#
#   print('---------')
    tfm_a.Translate(x[0], x[1], x[2])
    tfm_a.RotateWXYZ(x[3], x[4], x[5], x[6])
    coli.Update()
#
#   tmp=vtk.vtkTransform()
#   tmp.DeepCopy(tfm_a)
    vtp_a=tran(prt_a,tfm_a)
    bds_a=vtp_a.GetBounds()
    vtp_b=tran(prt_b,tfm_b)
    bds_b=vtp_b.GetBounds()
#
    c=coli.GetNumberOfContacts()
#   print('this',c)
    tfm_a.RotateWXYZ(-x[3], x[4], x[5], x[6])
    tfm_a.Translate(-x[0], -x[1], -x[2])
#   print('---------')
#
    bds=[0,0,0,0,0,0]
    for i in range(6):
        if i%2 == 0:
            bds[i] = min(bds_a[i],bds_b[i])
        else:
            bds[i] = max(bds_a[i],bds_b[i])
#
    f=c*1e6
    print(f)
    f=f+(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4]) #+ np.linalg.norm(x[:3])**2.
    print(bds_a)
    print(bds_b)
    print(bds,f)
    print('vol:',(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4]))
#
    return f
#
if __name__ == "__main__":
#
#   read in parts
#
    red = vtk.vtkSTLReader()
    red.SetFileName('./prt_0.stl')
    red.Update()
    prt_a = red.GetOutput()
#
    red = vtk.vtkSTLReader()
    red.SetFileName('./prt_1.stl')
    red.Update()
    prt_b = red.GetOutput()
#
#   get COMS, translate COM to 0,0,0, and keep the transform
#
    com = vtk.vtkCenterOfMass()
    com.SetInputData(prt_a)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    com_a = np.array(com.GetCenter())
#
    com = vtk.vtkCenterOfMass()
    com.SetInputData(prt_b)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    com_b = np.array(com.GetCenter())
#
    [prt_a,tfm_a_0,_] = move(prt_a,-com_a,[0.,1.,0.,0.])
    [prt_b,tfm_b_0,_] = move(prt_b,-com_b,[0.,1.,0.,0.])
#
#   set up transforms for collision filter, and filter itself
#
    [_,tfm_a,_] = move(prt_a,[0.,0.,0.],[0.,1.,0.,0.])
    [_,tfm_b,_] = move(prt_b,[0.,0.,0.],[0.,1.,0.,0.])
#
    coli = vtk.vtkCollisionDetectionFilter()
    coli.SetCollisionModeToFirstContact()
#   coli.SetBoxTolerance(100.)
#   coli.SetCellTolerance(1000.)
    coli.SetInputData(0, prt_a)
    coli.SetTransform(0, tfm_a)
    coli.SetInputData(1, prt_b)
    coli.SetTransform(1, tfm_b)
    coli.Update()
    c=coli.GetNumberOfContacts()
    print('There should be contact now', c)
#
    x0=np.zeros(7*1)
    x0[0]=1e1
    x0[4]=0e0
    x0[5]=0e0
    x0[6]=1e0
#   tmp=vtk.vtkTransform()
#   tmp.DeepCopy(tfm_a)
    simu(x0,coli,tfm_a,tfm_b,prt_a,prt_b)
    x0=np.zeros(7*1)
    x0[0]=-1e1
    x0[4]=0e0
    x0[5]=0e0
    x0[6]=1e0
    simu(x0,coli,tfm_a,tfm_b,prt_a,prt_b)
#
    l=[-1e3]*7*1
    u=[1e3]*7*1
#
#   simu(x0,coli,tfm_a,tfm_b)
#   stop
    res=differential_evolution(simu,args=(coli,tfm_a,tfm_b,prt_a,prt_b),callback=conv,bounds=list(zip(l,u)),workers=1,seed=69,polish=True,maxiter=10000, init='sobol')
#   res=differential_evolution(simu,args=(coli,tfm_a,tfm_b),bounds=list(zip(l,u)),workers=-1,seed=1)
    print(res)
    x=res.x
#   [prt_a,_,_] = move(prt_a,x[:3],x[3:7])
    simu(x,coli,tfm_a,tfm_b,prt_a,prt_b)
#   wout(prt_a,'prt_a',0)
#   cfm_0=coli.GetTransform(0)
#   cfm_1=coli.GetTransform(1)
#   print(cfm_0)
#   print(cfm_1)
    [vtp,_,_] = move(prt_a,x[:3],x[3:7])
#   vtp = tran(prt_a,tfm_)
    wout(vtp,'see',0)
#
