#
from scipy.spatial.transform import Rotation as R

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
from vtk.util import numpy_support
#
from simu_ga import simu_ga, back_ga, back_sa
from simu_ga_na import simu_ga_na, back_ga_na
from simu_nm import simu_nm, back_nm
from simu_nm_co import simu_nm_co, back_nm_co
from simu_bp2 import simu_bp, back_bp, back_bp2, back_bp3
#
from util import tran
from util import move
from util import appd
from util import woutfle
from util import woutstr
#
class Part:
#
    def __init__(self,idn,obj,tfm):
        self.idn=idn
        self.obj=obj
        self.tfm=tfm
#
class Object:
#
    def __init__(self,idn,vtp_0,stp_0,cen_0,vtp,stp,cen,vtc,stc,pts):
        self.idn=idn
        self.vtp_0=vtp_0
        self.stp_0=stp_0
        self.cen_0=cen_0
        self.vtp=vtp
        self.stp=stp
        self.cen=cen
        self.vtc=vtc
        self.stc=stc
        self.pts=pts
#
def Data(lst):
#
    tmp0=[]
    tmp1=[]
    tmp2=[]
    tmp3=[]
    tmp4=[]
    for i in lst:
        tmp0.append(i.pts)
        tmp1.append(i.stp)
        tmp2.append(i.vtp)
        tmp3.append(i.stc)
        tmp4.append(i.vtc)
    return tmp0,tmp1,tmp2,tmp3,tmp4
#
if __name__ == "__main__":
#
#   parameters
#
    c_a=180.
    c_v=0.
    c_l=-1 # estimated if negative
#
#   get input arguments 
#   - number to be stacked and
#   - path to file
#
    c=0
    nums=[]; flns=[]
    while True:
        try:
            nums.append(int(sys.argv[c+1]))
            flns.append(sys.argv[c+2])
            c=c+2
        except: break
#
    nobj=int(c/2) # number of unique parts (objects)
#
    n=0 
    js_str=[]; objs_bbv=[]; objs_vtp=[]
    flns_str=[]; objs_cub=[]
    cubs_str=[]; objs_pts=[]
#
    objs=[]; tfms=[]; maps=[]
    cols=[]
#
#   for each object
#
    for i in range(nobj):
#
#       obj=Object()
#
        print('='*40)
        print('Pack %2d of %10s: '%(nums[i],flns[i]))
        print('-'*40)
#
#       read the stl into polydata
#
        flt=vtk.vtkSTLReader()
        flt.SetFileName(flns[i])
        flt.Update()
        obj_vtp_0=flt.GetOutput()
        obj_stp_0=woutstr(obj_vtp_0) # file content string; for parallelization
#
#       properties
#
        flt=vtk.vtkCenterOfMass()
        flt.SetInputData(obj_vtp_0)
        flt.SetUseScalarsAsWeights(False)
        flt.Update()
        obj_cen_0=flt.GetCenter()
#
        prp = vtk.vtkMassProperties()
        prp.SetInputData(obj_vtp_0)
        prp.Update() 
        obj_vol_0=prp.GetVolume()
        obj_are_0=prp.GetSurfaceArea()
#
        print('-points   : %14d'%obj_vtp_0.GetNumberOfPoints())
        print('-cells    : %14d'%obj_vtp_0.GetNumberOfCells())
        print('-centroid : %14.7e'%(obj_cen_0[0]))
        print('          : %14.7e'%(obj_cen_0[1]))
        print('          : %14.7e'%(obj_cen_0[2]))
        print("-area     : %14.7e"%obj_are_0)
        print("-volume   : %14.7e"%obj_vol_0)
#
#       clean it
#
        flt=vtk.vtkCleanPolyData()
        flt.SetInputData(obj_vtp_0)
        flt.SetTolerance(1e-4) # fraction of BB diagonal
        flt.Update()
        obj_vtp=flt.GetOutput()
#
#       only triangles
#
        flt=vtk.vtkTriangleFilter()
        flt.SetInputData(obj_vtp)
        flt.SetPassLines(False)
        flt.SetPassVerts(False)
        flt.Update()
        obj_vtp=flt.GetOutput()
#
        print('-'*40)
        print('Cleaned and decimated: ')
        print('-'*40)
#
        if obj_vtp.GetNumberOfCells() > 200:
            flt=vtk.vtkQuadricDecimation()
            flt.SetInputData(obj_vtp)
            flt.SetTargetReduction((obj_vtp.GetNumberOfCells()-200)/obj_vtp.GetNumberOfCells())
            flt.SetVolumePreservation(True)
            flt.Update()
            obj_vtp=flt.GetOutput()
#
#       properties again
#
        flt=vtk.vtkCenterOfMass()
        flt.SetInputData(obj_vtp)
        flt.SetUseScalarsAsWeights(False)
        flt.Update()
        obj_cen=flt.GetCenter()
#
        prp = vtk.vtkMassProperties()
        prp.SetInputData(obj_vtp)
        prp.Update() 
        obj_vol=prp.GetVolume()
        obj_are=prp.GetSurfaceArea()
#
        print('-points   : %14d'%obj_vtp.GetNumberOfPoints())
        print('-cells    : %14d'%obj_vtp.GetNumberOfCells())
        print('-centroid : %14.7e'%(obj_cen[0]))
        print('          : %14.7e'%(obj_cen[1]))
        print('          : %14.7e'%(obj_cen[2]))
        print("-area     : %14.7e"%obj_are)
        print("-volume   : %14.7e"%obj_vol)
#
#       transform updated object so that its centroid is 0,0,0
#
        tfm_0=vtk.vtkTransform()
        tfm_0.Translate(-obj_cen[0], -obj_cen[1], -obj_cen[2])
        tfm_0.Update()
        flt=vtk.vtkTransformPolyDataFilter()
        flt.SetInputData(obj_vtp)
        flt.SetTransform(tfm_0)
        flt.Update()
        obj_vtp=flt.GetOutput()
        obj_stp=woutstr(obj_vtp)
#
#       get axis aligned bounds and bounding box volume
#
        obj_bds=obj_vtp.GetBounds()
        obj_bbv=(obj_bds[1]-obj_bds[0])*(obj_bds[3]-obj_bds[2])*(obj_bds[5]-obj_bds[4])
#
        print("-axis aligned bounding box: %14.7e"%(obj_bds[1]-obj_bds[0]))
        print("                          : %14.7e"%(obj_bds[3]-obj_bds[2]))
        print("                          : %14.7e"%(obj_bds[5]-obj_bds[4]))
        print("-with volume: %14.7e"%obj_bbv)
#
#       make cube source
#
        src=vtk.vtkCubeSource()
        src.SetBounds(obj_bds)
        src.Update()
        obj_vtc=src.GetOutput()
#
#       clean it
#
        flt=vtk.vtkCleanPolyData()
        flt.SetInputData(obj_vtc)
        flt.SetTolerance(1e-4) # fraction of BB diagonal
        flt.Update()
        obj_vtc=flt.GetOutput()
#
#       triangles
#
        flt=vtk.vtkTriangleFilter()
        flt.SetInputData(obj_vtc)
        flt.Update()
        obj_vtc=flt.GetOutput()
        obj_stc=woutstr(obj_vtc)
#
#       get bounding box points
#
        obj_pts=numpy_support.vtk_to_numpy(obj_vtc.GetPoints().GetData())
#
#       make the object
#       
        obj=Object(i,obj_vtp_0,obj_stp_0,obj_cen_0,obj_vtp,obj_stp,obj_cen,obj_vtc,obj_stc,obj_pts)
        objs.append(obj)
#
#       make some things for each _part_
#       add up some things
#
        for j in range(nums[i]):
#
#           make a transform
#
            tfm=vtk.vtkTransform()
            tfm.Translate(0., 0., 0.)
            tfm.Update()
#
            tfms.append(tfm)
            maps.append(i)
#
            c_v=c_v+obj_bbv
            n=n+1
#
#   ....
#
    print(c_v)
#
    opt_0_bds=[[-1.,1.] for i in range(4*n)]; 
    opt_0_its=[False for i in range(4*n)]
    for c in range(n):
        opt_0_bds[4*c+3]=[-3,3]
        opt_0_its[4*c+3]=True
    opt_0_bds=tuple(opt_0_bds)
    opt_0_its=tuple(opt_0_its)
#
    opt_1_bds=[[-1.,1.] for i in range(7*n)]; 
    opt_1_its=[False for i in range(7*n)]
    opt_1_bds=tuple(opt_1_bds)
    opt_1_its=tuple(opt_1_its)
#
    [pnts,stps,vtps,stcs,vtcs]=Data(objs)
#
    cols=[]
    m = int(n*(n-1)/2) # number of pairs (combination formula)
    print('='*40)
    print('Making %d collision objects'%m)
    print('-'*40)
    for i in range(n-1):
        for j in range(i+1,n):
#
            col=vtk.vtkCollisionDetectionFilter()
            col.SetCollisionModeToAllContacts()
#           col.SetCollisionModeToHalfContacts()
#           col.SetCollisionModeToFirstContact()
            col.SetInputData(0,vtps[maps[i]])
            col.SetTransform(0,tfms[i])
            col.SetInputData(1,vtps[maps[j]])
            col.SetTransform(1,tfms[j])
            col.Update()
            cols.append(col)
#
    c_l=np.array([100.,100.,100.])#  for in box
    c_a=180
#
    c_r=[]
    r=R.from_rotvec(0 * np.array([1.,1.,1.])).as_matrix().T 
    c_r.append(r)
    r=R.from_rotvec(2*np.pi/3 * np.array([1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(np.pi/2 * np.array([1,0,0])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(np.pi/2 * np.array([0,1,0])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(np.pi/2 * np.array([0,0,1])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(2*np.pi/3 * np.array([-1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(0 * np.array([1.,1.,1.])).as_matrix().T 
    c_r.append(r)
#
    res=dual_annealing(simu_bp,args=(n,pnts,maps,c_l,c_r,c_v,0),bounds=opt_0_bds,\
        callback=partial(back_bp3,args=(n,pnts,maps,c_l,c_r,c_v,nums,stps,stcs)),
        seed=1,no_local_search=True,maxiter=1000)
#
    xi=np.zeros(7*n)
    x=res.x
    for i in range(n):
        xi[7*i+0]=x[4*i+0]
        xi[7*i+1]=x[4*i+1]
        xi[7*i+2]=x[4*i+2]
#
        if x[i*4+3] >= 0-3.5 and x[i*4+3] < 1-3.5:
            r=R.from_matrix(c_r[0].T).as_rotvec()
        elif x[i*4+3] >= 1-3.5 and x[i*4+3] < 2-3.5:
            r=R.from_matrix(c_r[1].T).as_rotvec()
        elif x[i*4+3] >= 2-3.5 and x[i*4+3] < 3-3.5:
            r=R.from_matrix(c_r[2].T).as_rotvec()
        elif x[i*4+3] >= 3-3.5 and x[i*4+3] < 4-3.5:
            r=R.from_matrix(c_r[3].T).as_rotvec()
        elif x[i*4+3] >= 4-3.5 and x[i*4+3] < 5-3.5:
            r=R.from_matrix(c_r[4].T).as_rotvec()
        elif x[i*4+3] >= 5-3.5 and x[i*4+3] < 6-3.5:
            r=R.from_matrix(c_r[5].T).as_rotvec()
        elif x[i*4+3] >= 6-3.5 and x[i*4+3] < 7-3.5:
            r=R.from_matrix(c_r[6].T).as_rotvec()
#
        tmp=np.linalg.norm(r)
        t=np.array([np.rad2deg(tmp)/c_a, r[0]/max(tmp,1e-9),r[1]/max(tmp,1e-9),r[2]/max(tmp,1e-9)])
        xi[7*i+3]=t[0]
        xi[7*i+4]=t[1]
        xi[7*i+5]=t[2]
        xi[7*i+6]=t[3]
#
    print(res)
#
    res=dual_annealing(simu_nm_co,args=(n,cols,tfms,vtps,maps,c_l,c_a,c_v,0),bounds=opt_1_bds,\
        callback=partial(back_nm_co,args=(n,cols,tfms,vtps,maps,c_l,c_a,c_v,nums,stps,stcs)),
        seed=1,no_local_search=True,maxiter=1000,x0=xi)#,workers=4,seed=1
#
    print(res)
#
    back_nm_co(res.x,0,0,[n,cols,tfms,vtps,maps,c_l,c_a,c_v,nums,stps,stcs])
#
    stop
#       callback=partial(back_bp,args=(n,nobj,cubs_str,objs_num,c_l,c_a,c_v,cols,prts_tfm,prts_map,objs_str,objs_cub, objs_pts)),
#       seed=1,maxiter=1,polish=True,disp=True,popsize=1)#,integrality=tup_its,popsize=1)
#   print(res)
#   stop
#   res=differential_evolution(simu_bp,args=(n,cols,prts_tfm,objs_cub,prts_map,c_l,c_a,c_v,0),\
#       bounds=tup_bds,
#       seed=1,maxiter=10000,workers=1,polish=True,disp=True,integrality=tup_its)#,x0=xi)#,workers=4,seed=1
#   res=differential_evolution(simu_bp,args=(n,cols,prts_tfm,objs_vtp,prts_map,c_l,c_a,c_v,0),\
#       bounds=tup_bds,callback=partial(back_bp,args=(n,cols,prts_tfm,objs_vtp,prts_map,c_l,c_a,c_v,nobj,objs_str,objs_num)),
#       seed=1,maxiter=10000,workers=1,polish=True,disp=True,integrality=tup_its)#,x0=xi)#,workers=4,seed=1
    xi=res.x
    res=minimize(simu_bp,args=(n,cols,prts_tfm,objs_cub,objs_pts,prts_map,c_l,c_a,c_v,0), x0=xi, bounds=tup_bds, method='Nelder-Mead',\
        options={'disp': True, 'adaptive': True, 'fatol':1e-1},\
        callback=partial(back_bp2,args=(n,nobj,cubs_str,objs_num,c_l,c_a,c_v,cols,prts_tfm,prts_map,objs_str,objs_cub, objs_pts)))
    print(res)
    stop
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
    c=0
    for i in range(n-1):
        for j in range(i+1,n):
#
            coli = vtk.vtkCollisionDetectionFilter()
            coli.SetCollisionModeToAllContacts()
#           coli.SetCollisionModeToHalfContacts()
#           coli.SetCollisionModeToFirstContact()
            coli.SetInputData(0, objs_vtp[prts_map[i]])
            coli.SetTransform(0, prts_tfm[i])
            coli.SetInputData(1, objs_vtp[prts_map[j]])
            coli.SetTransform(1, prts_tfm[j])
            coli.Update()
            cols.append(coli)
            c=c+1
    print('here here')
#
    bds=[[-1.,1.] for i in range(7*n)]; tup_bds=tuple(bds)
    its=[False for i in range(7*n)]
    for c in range(n):
        its[7*c+3]=True
        its[7*c+4]=True
        its[7*c+5]=True
        its[7*c+6]=True
    tup_its=tuple(its)
#
#
#   res=differential_evolution(simu_ga_na,args=(n,objs_str,c_l,c_a,c_v,0),\
#       workers=4,seed=1,polish=False,disp=True,maxiter=1,updating='deferred',\
#       callback=partial(back_ga_na,args=(n,objs_str,c_l,c_a,c_v)),bounds=tup_bds)#list(zip(l,u)))#,x0=xi)
    res=differential_evolution(simu_nm_co,args=(n,cols,prts_tfm,objs_vtp,prts_map,c_l,c_a,c_v,0),bounds=tup_bds,\
        callback=partial(back_ga,args=(n,nobj,objs_str,objs_num,c_l,c_a,c_v,objs_str)),
        seed=1,maxiter=100,polish=False,disp=True,integrality=tup_its)#,x0=xi)#,workers=4,seed=1
#
    print(res)
    stop
#
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
            callback=partial(back_nm_co,args=(n,cols,tfms,objs_vtp,prts_map,c_l,c_a,c_v)))
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
