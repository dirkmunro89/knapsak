#
import os
import sys
import vtk
import time
import numpy as np
import logging as log
from functools import partial
from vtk.util import numpy_support
from scipy.optimize import minimize, dual_annealing
#
#
from rndr import rndr
#
from init import init
from init import pretfms12 as pretfms
from simu_obp import simu_obp, back_da
from simu_obp_co import simu_obp_co, back_da_co
#
from util import tfmx, tran, appdata, woutfle
#
if __name__ == "__main__":
#
#   parameters
#
    c_l=np.array([200.,200.,200.]) # for in box
    c_s=1.01
    c_a=180
    c_e=1000
#
#   get input arguments 
#   - number to be stacked and
#   - path to file
#
    t0=time.time()
    out='./out_%d/'%t0
    if not os.path.isdir(out):
        os.makedirs(out)
    else:
        print('error')
        sys.exit(1)
#
    level=log.INFO
    format   = '%(message)s'
    handlers=[log.FileHandler('history_%d.log'%t0), log.StreamHandler()]
    log.basicConfig(level=level, format=format, handlers=handlers)
#
#
    log.info('='*60)
    tmp=" ".join(sys.argv)
    c=0
    while True:
        try: tmp[c+60]
        except: break
        log.info(tmp[c:c+60])
        c=c+60
#
    log.info(tmp[c:])
    log.info('-'*60)
    log.info('Writing output to:\n%s'%out)
    log.info('='*60)
#
    sys.argv=['prerot.py', 'objsix', '0', '7', 'stl/Plate.stl']
#
    opt_str=sys.argv[1]
    vis_flg=int(sys.argv[2])
#
    c=0
    nums=[]; flns=[]
    while True:
        try: sys.argv[c+3]
        except: break
        nums.append(int(sys.argv[c+3]))
        flns.append(sys.argv[c+4])
        c=c+2
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
        log.info('='*60)
        log.info('Pack %2d of %10s: '%(nums[i],flns[i]))
        log.info('-'*60)
#
#       make the object from the input file
#
        obj=init(i,flns[i],c_e,c_s,log)
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
    log.info('='*60)
    log.info('Total volume of AABB volumes : %7.3e'%(c_v_0))
    log.info('Total volume                 : %7.3e'%(c_v_1))
    log.info('Efficiency                   : %7.3e'%(c_v_1/c_v_0))
    log.info('='*60)
#
    cols=[]
    m = int(n*(n-1)/2) # number of pairs (combination formula)
#
    log.info('Making %d collision objects'%m)
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
    log.info('='*60)
#
    pnts = [obj.pts for obj in objs]
    stps = [obj.stp for obj in objs]
    vtps = [obj.vtp for obj in objs]
    stcs = [obj.stc for obj in objs]
    vtcs = [obj.vtc for obj in objs]
    exts = [obj.ext for obj in objs]
#
#   set up predefined transforms
#
    c_r=pretfms()
#
#   set bounds and integer variables
#
    opt_0_x=np.array([0 for i in range(4*n)])
    opt_0_bds=[(-1.,1.) for i in range(4*n)]; 
    for c in range(n):
        opt_0_bds[4*c]=(-1,1)
    opt_0_bds=tuple(opt_0_bds)
#
    opt_1_bds=[(-1.,1.) for i in range(7*n)]; 
    opt_1_x=np.array([0 for i in range(7*n)])
    opt_1_bds=tuple(opt_1_bds)
#
    if 'obj' in opt_str:
        log.info('%6s%15s%15s%16s'%('k','F_0 (opt)','F_0 (sim)','collisions'))
    else:
        log.info('%14s%16s'%('F_0','collisions'))
    log.info('-'*60)
#
    outs_0=[]
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
        for j in range(nums[i]):
            tmp=[0.]*16
            tfm.GetMatrix().DeepCopy(tmp,tfm.GetMatrix())
            outs_0.append(tmp)
#
    if 'sox' in opt_str:
        app=appdata(opt_1_x,n,nums,maps,vtps_0,c_l,c_a,c_r,2,0,1)
    elif 'six' in opt_str:
        app=appdata(opt_0_x,n,nums,maps,vtps_0,c_l,c_a,c_r,1,0,1)
    else:
        app=appdata(opt_1_x,n,nums,maps,vtps_0,c_l,c_a,c_r,0,0,1)
#
    if vis_flg:
        vis=rndr(app)
    else:
        vis=None
#
    xk=np.zeros(4*n)
    xk[4*0]=-3/3.
    xk[4*1]=-2/3.
    xk[4*2]=-1/3.
    xk[4*3]=0/3.
    xk[4*4]=1/3.
    xk[4*5]=2/3.
    xk[4*6]=3/3.
#
    for i in range(n):
# 
        if 'sox' in opt_str:
            tfm=tfmx(xk,i,c_l,c_a,c_r,None,2,0)
            tmp=tran(vtps_0[maps[i]],tfm)
            woutfle(out,tmp,'build',-i-1)
        elif 'six' in opt_str:
            tfm=tfmx(xk,i,c_l,c_a,c_r,None,1,0)
            tmp=tran(vtps_0[maps[i]],tfm)
            woutfle("./",tmp,'build',-i-1)
        else:
            tfm=tfmx(xk,i,c_l,c_a,c_r,None,0,0)
            tmp=tran(vtps_0[maps[i]],tfm)
            woutfle(out,tmp,'build',-i-1)
#
        tmp=[0.]*16
        tfm.GetMatrix().DeepCopy(tmp,tfm.GetMatrix())
        outs_0[i].extend(tmp)
#
    if 'sox' in opt_str:
        app=appdata(xk,n,nums,maps,vtps_0,c_l,c_a,c_r,2,0,1)
        woutfle("./",app.GetOutput(),'build',0)
        app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,2,0,1)
        woutfle(out,app.GetOutput(),'objec',0)
        app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,2,0,1)
        woutfle(out,app.GetOutput(),'cubes',0)
    elif 'six' in opt_str:
        app=appdata(xk,n,nums,maps,vtps_0,c_l,c_a,c_r,1,0,1)
        woutfle("./",app.GetOutput(),'build',0)
        app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,1,0,1)
        woutfle(out,app.GetOutput(),'objec',0)
        app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,1,0,1)
        woutfle(out,app.GetOutput(),'cubes',0)
    else:
        app=appdata(xk,n,nums,maps,vtps_0,c_l,c_a,c_r,0,0,1)
        woutfle(out,app.GetOutput(),'build',0)
        app=appdata(xk,n,nums,maps,vtps,c_l,c_a,c_r,0,0,1)
        woutfle(out,app.GetOutput(),'objec',0)
        app=appdata(xk,n,nums,maps,vtcs,c_l,c_a,c_r,0,0,1)
        woutfle(out,app.GetOutput(),'cubes',0)
#
    with open(out+'transforms.dat', 'w') as file:
        file.writelines('\t'.join(str(j) for j in i) + '\n' for i in outs_0)
#
    log.info('-'*60)
    log.info('Result written to build.vtp and transforms.dat')
    log.info('Load build.vtp in Paraview or with: python load.py build.vtp')
    log.info('='*60)
#
    t1=time.time()
    log.info('Time taken (s): %14.7e'%(t1-t0))
    log.info('='*60)
#
