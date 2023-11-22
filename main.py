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
    print('='*40)
    print('Total volume of AABB volumes : %7.3e'%(c_v_0))
    print('Total volume                 : %7.3e'%(c_v_1))
    print('Efficiency                   : %7.3e'%(c_v_1/c_v_0))
    print('='*40)
#
    cols=[]
    m = int(n*(n-1)/2) # number of pairs (combination formula)
#
    print('Making %d collision objects'%m)
    print('-'*40)
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
    if meth == 'objall':
#
#       dual annealing full collisions (based on objects) continuous rotations
#
        res=dual_annealing(simu_obp_co,args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,0,0),\
            bounds=opt_1_bds,seed=1,no_local_search=False,\
            callback=partial(back_da_co,args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,\
            nums,vtps,vtcs,0,0)),maxiter=1000)
#
    elif meth == 'objsix':
#
#       dual annealing full collisions (based on objects) 6 rotations
#
        res=dual_annealing(simu_obp_co,args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,1,0),\
            bounds=opt_0_bds,seed=1,no_local_search=False,\
            callback=partial(back_da_co,args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,\
            nums,vtps,vtcs,1,0)))
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
        res=basinhopping(simu_obp,minimizer_kwargs={"method":"Nelder-Mead","args":(n,pnts,maps,c_l,c_r,c_v_0,0)},\
            callback=partial(back_bh,args=(n,pnts,maps,c_l,c_r,c_v_0,nums,vtps,vtcs,1,0)),\
            seed=1,x0=xi)
#
    print(res)
#
    x=res.x
#
    vtps_0=[]
    for i in range(n):
#
        tfm=vtk.vtkTransform()
#
#       translate input STL to the same center of rotation as the cleaned and decimated STL
#
        tfm.Translate(-objs[maps[i]].cen[0],-objs[maps[i]].cen[1],-objs[maps[i]].cen[2])
        tfm.Update()
        vtps_0.append(tran(objs[maps[i]].vtp_0,tfm))
#
    if 'six' in meth:
        app=appdata(x,n,nums,maps,vtps_0,c_l,c_a,c_r,1,0)
        woutfle(app.GetOutput(),'build',1)
    else:
        app=appdata(x,n,nums,maps,vtps_0,c_l,c_a,c_r,0,0)
        woutfle(app.GetOutput(),'build',1)
        app=appdata(x,n,nums,maps,vtps,c_l,c_a,c_r,0,0)
        woutfle(app.GetOutput(),'objec',1)
#
    stop
#
    xi=np.zeros(7*n)
    x=res.x
    for i in range(n):
        xi[7*i]=x[4*i]
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
    res=dual_annealing(simu_nm_co,args=(n,cols,tfms,vtps,maps,c_l,c_a,c_v_1,0),bounds=opt_1_bds,\
        callback=partial(back_nm_co,args=(n,cols,tfms,vtps,maps,c_l,c_a,c_v,nums,stps,stcs)),
        seed=1,no_local_search=True,maxiter=100,x0=xi,initial_temp=1.)#,workers=4,seed=1
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
