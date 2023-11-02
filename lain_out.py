#
import os
import sys
import vtk
import pickle
import numpy as np
import multiprocessing
from functools import partial
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
#
#
def conv2(xk,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu(xk,n,objs,c_l,c_a,c_v,1)
    print('%14.3e %6d'%(f,c),flush=True)
#
#   for i in range(n):
#
#       red = vtk.vtkXMLPolyDataReader()
#       red.ReadFromInputStringOn()
#       red.SetInputString(objs[i])
#       red.Update()
#       obj = red.GetOutput()
#
#       if i < n-1:
#           tmp = xk[7*i:7*i+7]
#       else:
#           tmp=np.zeros(7)
#           tmp[3:]=xk[7*i:7*i+4]
#       [vtp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
#       woutfle(vtp,'see',i)
#
    return False
#
def conv(xk,convergence,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu(xk,n,objs,c_l,c_a,c_v,1)
    print('%7.3f %14.3e %6d'%(convergence,f,c),flush=True)
#
    for i in range(n):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
#
#       if i < n-1:
        tmp = xk[7*i:7*i+7]
#       else:
#           tmp=np.zeros(7)
#           tmp[3:]=xk[7*i:7*i+4]
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
#       if i < n-1:
        tfm.Translate(c_l*x[i*7+0], c_l*x[i*7+1], c_l*x[i*7+2])
        tfm.RotateWXYZ(c_a*x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
#       else:
#           tfm.RotateWXYZ(c_a*x[i*7+0], x[i*7+1], x[i*7+2], x[i*7+3])
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
    for i in range(n-1):
        for j in range(i+1,n):
#
            coli = vtk.vtkCollisionDetectionFilter()
            coli.SetCollisionModeToAllContacts()
#           coli.SetCollisionModeToHalfContacts()
#           coli.SetCollisionModeToFirstContact()
            coli.SetInputData(0, objs[i])
            coli.SetTransform(0, tfms[i])
            coli.SetInputData(1, objs[j])
            coli.SetTransform(1, tfms[j])
            coli.Update()
            c=c+coli.GetNumberOfContacts()
#
    f=bds[5]#-bds[4])# + np.linalg.norm(x[:3])**2.
#   f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])# + np.linalg.norm(x[:3])**2.
#
#   f=f/c_v 
    f=f+c*c_v/n/309 #est. volume per triangle
#
    b=0.
    ext=100.
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
    if bds[4]<0.:
        b=b+(abs(bds[4]))**1.
        y=y+1
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
    c_a=180.0*2
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
    l=[-1e0]*(7*n)
    u=[1e0]*(7*n)
#
    res=differential_evolution(simu,args=(n,objs,c_l,c_a,c_v,0),callback=partial(conv,args=(n,objs,c_l,c_a,c_v)),bounds=list(zip(l,u)),workers=4,seed=1,polish=False,disp=True,maxiter=10,updating='deferred')
#
    x0=res.x
    bds=[[-1.,1.] for i in range(7*n)]; tup_bds=tuple(bds)
    print(res,flush=True)
    x=res.x
    for i in range(n):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
#
#       if i < n-1:
        tmp = x[7*i:7*i+7]
#       else:
#           tmp=np.zeros(7)
#           tmp[3:]=x[7*i:7*i+4]
        [vtp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        woutfle(vtp,'see',i)
#
#
    f=1e80
    fold=0.
    while abs(f-fold)>0.1:
        fold=f
        res=minimize(simu,args=(n,objs,c_l,c_a,c_v,0), x0=x0, bounds=tup_bds, method='Nelder-Mead', options={'disp': True, 'adaptive': True, 'fatol':1e-2},callback=partial(conv2,args=(n,objs,c_l,c_a,c_v)))#, 'eps': 1e-32, 'gtol': 1e-32, 'tol': 1e-32, 'xatol': 1e-16, 'fatol': 1e-16})
        print(res,flush=True)
        x0=res.x
        f=res.fun
        print(f,x,flush=True)
#   res=minimize(simu,args=(n,objs,c_l,c_a,c_v,0), x0=x0, bounds=tup_bds, method='Nelder-Mead', options={'disp': True,'maxfev': 1000000, 'maxiter': 1000000})#, 'eps': 1e-32, 'gtol': 1e-32, 'tol': 1e-32, 'xatol': 1e-16, 'fatol': 1e-16})
#
#   print(res)
#   x0=res.x
#   res=minimize(simu,args=(n,objs,c_l,c_a,c_v,0), x0=x0, bounds=tup_bds, method='Nelder-Mead', options={'disp': True,'maxfev': 1000000, 'maxiter': 1000000})#, 'eps': 1e-32, 'gtol': 1e-32, 'tol': 1e-32, 'xatol': 1e-16, 'fatol': 1e-16})
#   print(res)
#   x=res.x
#   res=minimize(simu,args=(n,objs,c_l,c_a,c_v,0), x0=x0, bounds=tup_bds, method='Nelder-Mead', options={'disp': True,'maxfev': 1000000, 'maxiter': 1000000})#, 'eps': 1e-32, 'gtol': 1e-32, 'tol': 1e-32, 'xatol': 1e-16, 'fatol': 1e-16})
#   print(res)
#   x=res.x
        for i in range(n):
#
            red = vtk.vtkXMLPolyDataReader()
            red.ReadFromInputStringOn()
            red.SetInputString(objs[i])
            red.Update()
            obj = red.GetOutput()
#
#           if i < n-1:
            tmp = x0[7*i:7*i+7]
#           else:
#               tmp=np.zeros(7)
#               tmp[3:]=x0[7*i:7*i+4]
            [vtp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
            woutfle(vtp,'lee',i)
    stop
#
