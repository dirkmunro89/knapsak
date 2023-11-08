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
from simu_ga import simu_ga, back_ga, back_sa
from simu_ga_na import simu_ga_na, back_ga_na
from simu_nm import simu_nm, back_nm
from simu_nm_co import simu_nm_co, back_nm_co
#
from util import tran
from util import move
from util import appd
from util import woutfle
from util import woutstr
#
if __name__ == "__main__":
#
#   parameters
#
    c_a=180.
    c_v=0.
#
#   get input arguments (number of each part to be stacked)
#
    c=0
    objs_num=[]; objs_fln=[]
    while True:
        try:
            objs_num.append(int(sys.argv[c+1]))
            objs_fln.append(sys.argv[c+2])
            c=c+2
        except: break
#
    nobj=int(c/2) # number of unique parts
#
    print('-'*80)
    n=0 
    objs_str=[]; objs_bbv=[]; objs_vtp=[]
    objs_tfm=[]; objs_map=[]
    for i in range(nobj):
#
        print('Stack %6d of %10s'%(objs_num[i],objs_fln[i]))
#
        red = vtk.vtkSTLReader()
        red.SetFileName(objs_fln[i])
        red.Update()
        obj_vtp = red.GetOutput()
#
        mrg = vtk.vtkCleanPolyData()
        mrg.SetInputData(obj_vtp)
        mrg.SetTolerance(1e-4) # fraction of BB diagonal
        mrg.Update()
        obj_vtp=mrg.GetOutput()
#
        alph=(obj_vtp.GetNumberOfCells()-1000)/obj_vtp.GetNumberOfCells()
        tmp=vtk.vtkPolyData()
        tmp.DeepCopy(obj_vtp)
#
        while tmp.GetNumberOfCells() > 1000:
#
            dec = vtk.vtkQuadricDecimation()
            dec.SetInputData(obj_vtp)
            dec.SetTargetReduction(alph)
            dec.SetVolumePreservation(True)
            dec.Update()
            tmp=dec.GetOutput()
#
        print('Triangles: %d'%tmp.GetNumberOfCells())
#
        obj_vtp.DeepCopy(tmp)
#
        prp = vtk.vtkMassProperties()
        prp.SetInputData(obj_vtp)
        prp.Update() 
#
        print("Volume = ", prp.GetVolume())
        print("Area = ", prp.GetSurfaceArea())
#
        com = vtk.vtkCenterOfMass()
        com.SetInputData(obj_vtp)
        com.SetUseScalarsAsWeights(False)
        com.Update()
        g = np.array(com.GetCenter())
        print("COM = ", g)
        bds=obj_vtp.GetBounds()
        bbv=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])
        c_v=max(c_v,bbv)
        objs_bbv.append(bbv)
        print("BBV = ", bbv)
#
        [obj_vtp,tfm_0,_] = move(obj_vtp,-g,[0.,1.,0.,0.],np.array([1.,1.,1.]),1.)
        obj_str=woutstr(obj_vtp)
        objs_vtp.append(obj_vtp)

        objs_str.append(obj_str)
#       objs_str.extend([obj_str]*prts_num[i])
#
        n=n+objs_num[i]
#
        for j in range(objs_num[i]):
            [_,tfm,_] = move(obj_vtp,[0.,0.,0.],[0.,0.,0.,1.],[1.,1.,1.],1.)
            objs_tfm.append(tfm)
            objs_map.append(i)
#
        print('-'*80)
#
#
#
#   L = np.cbrt(c_v*n)
#   print(L)
#
#   fopt=1e80
#   xopt=[0.,0.,0.]
#   for k in range(n):
#       for j in range(n):
#           for i in range(n):
#
#               nk=k+1
#               nj=j+1
#               ni=i+1
#
#               Lx=ni*(bds[1]-bds[0])
#               Ly=nj*(bds[3]-bds[2])
#               Lz=nk*(bds[5]-bds[4])
#
#               f = abs(Lx-L) + abs(Ly-L) + abs(Lz-L)
#
#               if ni*nj*nk>n:
#                   if f < fopt:
#                       fopt=f
#                       xopt=np.array([ni,nj,nk])
#
#   print(fopt)
#   print(xopt)
#   tmp=np.amax(xopt)
#
#   c_l=np.array([tmp*(bds[1]-bds[0]),tmp*(bds[3]-bds[2]),tmp*(bds[5]-bds[4])])#  for min vol.
    c_l=np.array([500.,500.,500.])#  for in box
#
    print(c_l)
#
#   c=0
    xi = np.zeros(7*n)
#   for k in range(xopt[2]):
#       for j in range(xopt[1]):
#           for i in range(xopt[0]):
#               xi[7*c+0]=-100./c_l[0] + i*(bds[1]-bds[0])/c_l[0]
#               xi[7*c+1]=-100./c_l[1] + j*(bds[3]-bds[2])/c_l[1]
#               xi[7*c+2]=-100./c_l[2] + k*(bds[5]-bds[4])/c_l[2]
#               c=c+1
#               if c == n:
#                   break
#           if c == n:
#               break
#       if c == n:
#           break
#
    l=[-1e0]*(7*n)
    u=[1e0]*(7*n)
#
#
#   make a little append function like this:
#
#   c=0
#   app = vtk.vtkAppendDataSets()
#   app.SetOutputDataSetType(0)
#   for i in range(nobj):
#       red = vtk.vtkXMLPolyDataReader()
#       red.ReadFromInputStringOn()
#       red.SetInputString(objs_str[i])
#       red.Update()
#       obj = red.GetOutput()
#       for j in range(objs_num[i]):
#           tmp = xi[7*c:7*c+7]
#           [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
#           app.AddInputData(tmp)
#           c=c+1
#   app.Update()
#
    app=appd(xi,nobj,objs_str,objs_num,c_l,c_a)
    woutfle(app.GetOutput(),'zer',-1)
#
    print('here')
#
    bds=[[-1.,1.] for i in range(7*n)]; tup_bds=tuple(bds)
#
#   simulated annealing
#
#
#   stop
#
#   genetic algorithm
#
#   res=differential_evolution(simu_ga_na,args=(n,objs_str,c_l,c_a,c_v,0),\
#       workers=4,seed=1,polish=False,disp=True,maxiter=1,updating='deferred',\
#       callback=partial(back_ga_na,args=(n,objs_str,c_l,c_a,c_v)),bounds=tup_bds)#list(zip(l,u)))#,x0=xi)
#   print(res)
#   stop
#
#   print(res,flush=True)
#   x0=res.x
#
#   app = vtk.vtkAppendDataSets()
#   app.SetOutputDataSetType(0)
#   for i in range(n):
#       red = vtk.vtkXMLPolyDataReader()
#       red.ReadFromInputStringOn()
#       red.SetInputString(objs[i])
#       red.Update()
#       obj = red.GetOutput()
#       tmp = x0[7*i:7*i+7]
#       [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
#       app.AddInputData(tmp)
#   app.Update()
#   woutfle(app.GetOutput(),'see',-1)
#
#   nelder mead simplex
#
#
    cols=[]
#   
    print('herehere')
    c=0
    for i in range(n-1):
        for j in range(i+1,n):
#
            coli = vtk.vtkCollisionDetectionFilter()
            coli.SetCollisionModeToAllContacts()
#           coli.SetCollisionModeToHalfContacts()
#           coli.SetCollisionModeToFirstContact()
            coli.SetInputData(0, objs_vtp[objs_map[i]])
            coli.SetTransform(0, objs_tfm[i])
            coli.SetInputData(1, objs_vtp[objs_map[j]])
            coli.SetTransform(1, objs_tfm[j])
            coli.Update()
            cols.append(coli)
            c=c+1
    print('here here')
#
#   res=dual_annealing(simu_ga,args=(n,objs_str,c_l,c_a,c_v,0),bounds=tup_bds,\
    res=dual_annealing(simu_nm_co,args=(n,cols,objs_tfm,objs_vtp,objs_map,c_l,c_a,c_v,0),bounds=tup_bds,\
        callback=partial(back_sa,args=(n,nobj,objs_str,objs_num,c_l,c_a,c_v)),
        seed=1,no_local_search=True)#,x0=xi)#,workers=4,seed=1
    stop
#
    f=1e80
    fold=0.
    print(xi)
    xi=np.zeros_like(xi)
    while abs(f-fold)>0.1:
        fold=f
        res=minimize(simu_nm_co,args=(n,cols,tfms,objs_vtp,c_l,c_a,c_v,0), x0=xi, bounds=tup_bds, method='Nelder-Mead',\
            options={'disp': True, 'adaptive': True, 'fatol':1e-1},\
            callback=partial(back_nm_co,args=(n,cols,tfms,objs_vtp,objs_map,c_l,c_a,c_v)))
#       res=minimize(simu_nm,args=(n,objs_str,c_l,c_a,c_v,0), x0=xi, bounds=tup_bds, method='Nelder-Mead',\
#           options={'disp': True, 'adaptive': True, 'fatol':1e-1},\
#           callback=partial(back_nm,args=(n,objs_str,c_l,c_a,c_v)))
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
            red.SetInputString(objs_str[i])
            red.Update()
            obj = red.GetOutput()
            tmp = x0[7*i:7*i+7]
            [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
            app.AddInputData(tmp)
        app.Update()
        woutfle(app.GetOutput(),'lee',-1)
#
