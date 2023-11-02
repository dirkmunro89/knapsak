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
def back_nm(xk,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu(xk,n,objs,c_l,c_a,c_v,1)
    print('%14.3e %6d'%(f,c),flush=True)
#
    return False
#
def back_ga(xk,convergence,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu(xk,n,objs,c_l,c_a,c_v,1)
    print('%7.3f %14.3e %6d'%(convergence,f,c),flush=True)
#
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(n):
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
        tmp = xk[7*i:7*i+7]
        [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        app.AddInputData(tmp)
    app.Update()
    woutfle(app.GetOutput(),'see',-1)
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
    apps=[]
#
    tfm_0 = vtk.vtkTransform()
    tfm_0.Translate(0., 0., 0.)
    tfm_0.Update()
#
    for i in range(n):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(string[i])
        red.Update()
        obj = red.GetOutput()
#
        tfm = vtk.vtkTransform()
        tfm.Translate(c_l*x[i*7+0], c_l*x[i*7+1], c_l*x[i*7+2])
        tfm.RotateWXYZ(c_a*x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
        tfm.Update()
#
        vtp=tran(obj,tfm)
        objs.append(vtp)
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
    for i in range(n-1):
        app = vtk.vtkAppendDataSets()
        app.SetOutputDataSetType(0)
        for j in range(i+1,n):
            app.AddInputData(objs[j])
            app.Update()
        apps.append(app.GetOutput())
#
    c=0
    for i in range(n-1):
#
        coli = vtk.vtkCollisionDetectionFilter()
        coli.SetCollisionModeToAllContacts()
#       coli.SetCollisionModeToHalfContacts()
#       coli.SetCollisionModeToFirstContact()
        coli.SetInputData(0, objs[i])
        coli.SetTransform(0, tfm_0)
        coli.SetInputData(1, apps[i])
        coli.SetTransform(1, tfm_0)
        coli.Update()
        c=c+coli.GetNumberOfContacts()
#
    f=bds[5]
#
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
#   read in parts
#
    c_l=0.
    c_a=180.0*2
    objs=[]
    volt=0.
    c_v=0.
#
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
    res=differential_evolution(simu,args=(n,objs,c_l,c_a,c_v,0),,\
        callback=partial(back_ga,args=(n,objs,c_l,c_a,c_v)),bounds=list(zip(l,u)),workers=1,seed=1,\
        polish=False,disp=True,maxiter=100,updating='deferred')
#
    bds=[[-1.,1.] for i in range(7*n)]; tup_bds=tuple(bds)
    print(res,flush=True)
    x0=res.x
#
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(n):
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
        tmp = x0[7*i:7*i+7]
        [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        app.AddInputData(tmp)
    app.Update()
    woutfle(app.GetOutput(),'see',-1)
#
    f=1e80
    fold=0.
    while abs(f-fold)>0.1:
        fold=f
        res=minimize(simu,args=(n,objs,c_l,c_a,c_v,0), x0=x0, bounds=tup_bds, method='Nelder-Mead',\
            options={'disp': True, 'adaptive': True, 'fatol':1e-2},\
            callback=partial(back_nm,args=(n,objs,c_l,c_a,c_v)))
        print(res,flush=True)
        x0=res.x
        f=res.fun
        print(f,x0,flush=True)
#
        app = vtk.vtkAppendDataSets()
        app.SetOutputDataSetType(0)
        for i in range(n):
            red = vtk.vtkXMLPolyDataReader()
            red.ReadFromInputStringOn()
            red.SetInputString(objs[i])
            red.Update()
            obj = red.GetOutput()
            tmp = x0[7*i:7*i+7]
            [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
            app.AddInputData(tmp)
        app.Update()
        woutfle(app.GetOutput(),'lee',-1)
#
    stop
#
