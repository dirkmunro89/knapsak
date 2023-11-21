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
#
from util import tran
from util import move
from util import appd
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
    objs=[]; tfms=[]; maps=[]
#
    n=0 
    c_v_0=0.
    c_v_1=0.
#
    for i in range(nobj):
#
        print('='*40)
        print('Pack %2d of %10s: '%(nums[i],flns[i]))
        print('-'*40)
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
    print('='*40)
    print('Total volume of AABB volumes : %7.3e'%(c_v_0))
    print('Total volume                 : %7.3e'%(c_v_1))
    print('Efficiency                   : %7.3e'%(c_v_1/c_v_0))
    print('='*40)
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
#   clean the simus; make parallel (appended data) version for full collision
#   make heuristic packing (SA paper)
#   Where and how to use Nelder mead? include it.
#   play with prusa slicer... maybe add a command line call
#   think about problem... in SLM height is a thing... in FDM as well, considering failure 
#   (chance to shift / to misalign)... 
#   I have seen it
#   time also seems depenendent on it
#
    if 1 == 1:
        res=dual_annealing(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_da,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,no_local_search=False)
#
    elif 1 == 2:
#
#       do with start. implement SA guy start
#
        res=dual_annealing(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0i_bds,\
            callback=partial(back_da,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,no_local_search=False)
    elif 2 == 1:
        res=differential_evolution(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_de,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,workers=cpus,updating='deferred',polish=False,disp=False,integrality=opt_0_its)
    elif 1 == 2:
        res=differential_evolution(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_de,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,workers=cpus,updating='deferred',polish=False,disp=False)
    elif 1 == 2:
        res=shgo(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_x,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            workers=cpus,options={'disp': True})
    elif 1 == 2:
        res=direct(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_x,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)))
    elif 1 == 1:
        res=brute(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),ranges=opt_0_bds,workers=cpus)
#           callback=partial(back_x,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)))
    else:
#
#       variables can not be bounded.... use modulo function... it has a derivative
#
        res=dual_annealing(simu_obp,args=(n,pnts,maps,c_l,c_r,c_v_0,0),bounds=opt_0_bds,\
            callback=partial(back_da,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),
            seed=1,maxiter=1,no_local_search=False)
        print('here')
        xi=res.x
        res=basinhopping(simu_obp,seed=1,xo=xi,\
            minimizer_kwargs={"method":"Nelder-Mead","args":(n,pnts,maps,c_l,c_r,c_v_0,0)},\
            callback=partial(back_bh,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)))
#
    print(res)
#
