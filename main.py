#
import os
import sys
import vtk
import math
import pickle
import numpy as np
import multiprocessing
#
from functools import partial
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import shgo
from scipy.optimize import brute
from scipy.optimize import direct
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
from vtk.util import numpy_support
#
from simu_ga import simu_ga, back_ga, back_sa
from simu_ga_na import simu_ga_na, back_ga_na
from simu_nm import simu_nm, back_nm
from simu_co import simu_co, back_co
#
from init import init, pretfms
#
from simu_obp import simu_obp, back_bp, back_de, back_da, back_x, back_bh
from simu_obp_co import simu_obp_co, back_da_co
#
from util import tran
from util import move
from util import appdata
from util import woutfle
from util import woutstr
#
# Note: 'object' and 'instance' is
#
if __name__ == "__main__":
#
#   get input arguments 
#   - number to be stacked and
#   - path to file
#
    cpus=4
    meth=sys.argv[1]
#
    c=0
    nums=[]; flns=[]
    while True:
        try:
            nums.append(int(sys.argv[c+2]))
            flns.append(sys.argv[c+3])
            c=c+2
        except: break
#
    nobj=int(c/2) # number of unique parts (objects)
#
    objs=[]; tfms=[]; maps=[]
#
    n=0 
    c_v_0=0.
    c_v_1=0.
#
    for i in range(nobj):
#
        print('='*60)
        print('Pack %2d of %10s: '%(nums[i],flns[i]))
        print('-'*60)
#
#       make the object from the input file
#
        obj=init(i,flns[i])
#
#       append to a list of the unique objects in the build
#
        objs.append(obj)
#
#       make a transform for each instance of the object in the build
#       and a list which maps back to the object id
#
        for j in range(nums[i]):
#
            tfm=vtk.vtkTransform()
            tfm.PostMultiply()
            tfm.Translate(0., 0., 0.)
            tfm.Update()
            tfms.append(tfm)
#
            maps.append(i)
#
            c_v_0=c_v_0+obj.bbv
            c_v_1=c_v_1+obj.vol
            n=n+1
#
    print('='*60)
    print('Total volume of AABB volumes : %7.3e'%(c_v_0))
    print('Total volume                 : %7.3e'%(c_v_1))
    print('Efficiency                   : %7.3e'%(c_v_1/c_v_0))
    print('='*60)
#
    cols=[]
    m = int(n*(n-1)/2) # number of pairs (combination formula)
#
    print('Making %d collision objects'%m)
    for i in range(n-1):
        for j in range(i+1,n):
#
            col=vtk.vtkCollisionDetectionFilter()
#           col.SetCollisionModeToAllContacts()
#           col.SetCollisionModeToHalfContacts()
            col.SetCollisionModeToFirstContact()
            col.SetInputData(0,objs[maps[i]].vtp)
            col.SetTransform(0,tfms[i])
            col.SetInputData(1,objs[maps[j]].vtp)
            col.SetTransform(1,tfms[j])
            col.Update()
            cols.append(col)
#
    print('='*60)
#
    pnts = [obj.pts for obj in objs]
    stps = [obj.stp for obj in objs]
    vtps = [obj.vtp for obj in objs]
    stcs = [obj.stc for obj in objs]
    vtcs = [obj.vtc for obj in objs]
#
#   parameters
#
    c_l=np.array([200.,200.,200.]) # for in box
    c_a=180
#
#   set up predefined transforms
#
    c_r=pretfms()
#
#   set bounds and integer variables
#
    opt_0_bds=[[-1.,1.] for i in range(4*n)]; 
    opt_0i_bds=[[-1e3,1e3] for i in range(4*n)]; 
    opt_0_its=[False for i in range(4*n)]
    for c in range(n):
        opt_0_bds[4*c]=[-3,3]
        opt_0_its[4*c]=True
    opt_0_bds=tuple(opt_0_bds)
    opt_0i_bds=tuple(opt_0i_bds)
    opt_0_its=tuple(opt_0_its)
#
    opt_1_bds=[[-1.,1.] for i in range(7*n)]; 
    opt_1_its=[False for i in range(7*n)]
    opt_1_bds=tuple(opt_1_bds)
    opt_1_its=tuple(opt_1_its)
#
    print('%14s%16s'%('F_0','collisions'))
    print('-'*60)
#
    if meth == 'objall':
#
#       dual annealing full collisions (based on objects) continuous rotations
#
        simu_args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,0,0)
        back_args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,nums,vtps,vtcs,0,0)
        res=dual_annealing(simu_obp_co,args=simu_args,bounds=opt_1_bds,seed=1,maxiter=1,\
            callback=partial(back_da_co,args=back_args))
#
    elif meth == 'objsix':
#
#       dual annealing full collisions (based on objects) 6 rotations
#
        res=dual_annealing(simu_obp_co,args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,1,0),\
            bounds=opt_0_bds,seed=1,no_local_search=False,\
            callback=partial(back_da_co,args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,\
            nums,vtps,vtcs,1,0)),maxiter=1)
#
    elif meth == 'boxsix':
#
#       dual annealing axis aligned bounding box based collisions with 6 rotations
#
        res=dual_annealing(simu_obp,args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,1,0),bounds=opt_0_bds,\
            callback=partial(back_da,args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,no_local_search=False)
#
    elif meth == 'boxall':
#
#       dual annealing axis aligned bounding box based collisions with all rotations
#
        res=dual_annealing(simu_obp,args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,0,0),bounds=opt_1_bds,\
            callback=partial(back_da,args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,nums,vtps,vtcs,0,0)),\
            seed=1,no_local_search=False)
#
    else:
        print('error')
#
    print('='*60)
    print('Scipy Output')
    print('-'*60)
#
    print(res)
#
    vtps_0 = [obj.vtp_0 for obj in objs]
    for i in range(nobj):
#
        tfm=vtk.vtkTransform()
#
#       translate input STL so that it has the same center of rotation 
#       as the cleaned and decimated STL
#
        tfm.Translate(-objs[i].cen[0],-objs[i].cen[1],-objs[i].cen[2])
        tfm.Update()
        vtps_0[i]=tran(vtps_0[i],tfm)
#
    if 'six' in meth:
        app=appdata(res.x,n,nums,maps,vtps_0,c_l,c_a,c_r,1,0)
        woutfle(app.GetOutput(),'build',-1)
#       app=appdata(res.x,n,nums,maps,vtps,c_l,c_a,c_r,1,0)
#       woutfle(app.GetOutput(),'objec',1)
    else:
        app=appdata(res.x,n,nums,maps,vtps_0,c_l,c_a,c_r,0,0)
        woutfle(app.GetOutput(),'build',-1)
#       app=appdata(res.x,n,nums,maps,vtps,c_l,c_a,c_r,0,0)
#       woutfle(app.GetOutput(),'objec',1)
#
    print('-'*60)
    print('Result written to build.vtp')
    print('='*60)
