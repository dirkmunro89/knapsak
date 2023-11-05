#
import os
import sys
import vtk
import math
import pickle
import numpy as np
import multiprocessing
from functools import partial
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
#
from simu_ga import simu_ga
from simu_nm import simu_nm
#
from util import tran
from util import move
from util import woutfle
from util import woutstr
#
def arrstrt(n,string,c_l,c_a):
#
    objs=[]
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
        bds=obj.GetBounds()
#
        bdsx=bds[1]-bds[0]
        bdsy=bds[3]-bds[2]
        bdsz=bds[5]-bds[4]
#
        maxx=math.floor(200./bdsx)
        maxy=math.floor(200./bdsy)
        maxz=math.floor(200./bdsz)
#
    shfx = (maxx*bdsx-200.)/2.
    shfy = (maxy*bdsy-200.)/2.
#
    x0=np.zeros(7*n)
    c=0
    for k in range(10):
        for j in range(maxy):
            for i in range(maxx):
                if c == n:
                    break
                x0[7*c+0]=(-100+bdsx/2+i*bdsx - shfx)/c_l[0]
                x0[7*c+1]=(-100+bdsy/2+j*bdsy - shfy)/c_l[1]
                x0[7*c+2]=(-bds[4] +k*bdsz)/c_l[2]
                c=c+1
#
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(n):
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(string[i])
        red.Update()
        obj = red.GetOutput()
        tmp = x0[7*i:7*i+7]
        [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        app.AddInputData(tmp)
    app.Update()
    woutfle(app.GetOutput(),'zer',-1)
#
    return x0
#
def back_nm(xk,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu_nm(xk,n,objs,c_l,c_a,c_v,1)
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
    [f,c]=simu_ga(xk,n,objs,c_l,c_a,c_v,1)
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
if __name__ == "__main__":
#
    n=int(sys.argv[1])
    fln=sys.argv[2]
#
#   read in parts
#
    c_a=180.
    c_v=0.
#
#   single part
#
    print('Read in %s'%fln)
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
    print("Area= ", prp.GetSurfaceArea())
#
    com = vtk.vtkCenterOfMass()
    com.SetInputData(obj)
    com.SetUseScalarsAsWeights(False)
    com.Update()
    g = np.array(com.GetCenter())
    bds=obj.GetBounds()
    c_v=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])
    [obj,tfm_0,_] = move(obj,-g,[0.,1.,0.,0.],np.array([1.,1.,1.]),1.)
    obj=woutstr(obj)
#
    objs=[obj]*n
#
#   1) fit parts in this box, and minimize the height
#   2) if I want to do minimum volume, how to do bounds... 
#   please minimise the volume of these 10 parts... again back to start position...
#
#   I have N parts, I want to make a equal sided box... 
#
    L = np.cbrt(c_v*n)
    print(L)
#
    fopt=1e80
    xopt=[0.,0.,0.]
    for k in range(n):
        for j in range(n):
            for i in range(n):
#
                nk=k+1
                nj=j+1
                ni=i+1
#
                Lx=ni*(bds[1]-bds[0])
                Ly=nj*(bds[3]-bds[2])
                Lz=nk*(bds[5]-bds[4])
#
                f = abs(Lx-L) + abs(Ly-L) + abs(Lz-L)
#
                if ni*nj*nk>n:
                    if f < fopt:
                        fopt=f
                        xopt=np.array([ni,nj,nk])
#
    print(fopt)
    print(xopt)
    tmp=np.amax(xopt)
#
#   nx = math.ceil(200./(bds[1]-bds[0]))
#   ny = math.ceil(200./(bds[3]-bds[2]))
#   nz = math.ceil(200./(bds[5]-bds[4]))
#
    c_l=np.array([100.,100.,100.])#  for in box
    c_l=np.array([tmp*(bds[1]-bds[0]),tmp*(bds[3]-bds[2]),tmp*(bds[5]-bds[4])])#  for min vol.
#
    print(c_l)
#
    c=0
    xi = np.zeros(7*n)
    for k in range(xopt[2]):
        for j in range(xopt[1]):
            for i in range(xopt[0]):
                xi[7*c+0]=-100./c_l[0] + i*(bds[1]-bds[0])/c_l[0]
                xi[7*c+1]=-100./c_l[1] + j*(bds[3]-bds[2])/c_l[1]
                xi[7*c+2]=-100./c_l[2] + k*(bds[5]-bds[4])/c_l[2]
#               if c%3 == 0:
#                   xi[7*c+3]=1.
#                   xi[7*c+4]=1.
#                   xi[7*c+5]=0
#                   xi[7*c+6]=0
                c=c+1
                if c == n:
                    break
            if c == n:
                break
        if c == n:
            break
#
    l=[-1e0]*(7*n)
    u=[1e0]*(7*n)
#
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(n):
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
        tmp = xi[7*i:7*i+7]
        [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        app.AddInputData(tmp)
    app.Update()
    woutfle(app.GetOutput(),'zer',-1)
#
#   genetic algorithm
#
#   res=dual_annealing(simu_ga,args=(n,objs,c_l,c_a,c_v,0),bounds=list(zip(l,u)),callback=partial(back_ga,args=(n,objs,c_l,c_a,c_v)))#,workers=4,seed=1
    bds=[[-1.,1.] for i in range(7*n)]; tup_bds=tuple(bds)
    res=differential_evolution(simu_ga,args=(n,objs,c_l,c_a,c_v,0),\
        workers=4,seed=1,polish=False,disp=True,maxiter=1,updating='deferred',\
        callback=partial(back_ga,args=(n,objs,c_l,c_a,c_v)),bounds=list(zip(l,u)))#,x0=xi)
    stop
#
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
#   nelder mead simplex
#
    f=1e80
    fold=0.
    while abs(f-fold)>0.1:
        fold=f
        res=minimize(simu_nm,args=(n,objs,c_l,c_a,c_v,0), x0=x0, bounds=tup_bds, method='Nelder-Mead',\
            options={'disp': True, 'adaptive': True, 'fatol':1e-1},\
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
