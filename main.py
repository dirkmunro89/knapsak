#
import os
import sys
import vtk
import numpy as np
import logging as log
from functools import partial
from vtk.util import numpy_support
from scipy.optimize import minimize, dual_annealing
#
level=log.INFO
format   = '%(message)s'
handlers = [log.FileHandler('history.log'), log.StreamHandler()]
log.basicConfig(level = level, format = format, handlers = handlers)
#
from rndr import rndr
#
from init import init, pretfms
from simu_obp import simu_obp, back_da
from simu_obp_co import simu_obp_co, back_da_co
#
from util import tfmx, tran, appdata, woutfle
#
if __name__ == "__main__":
#
#   get input arguments 
#   - number to be stacked and
#   - path to file
#
    meth_flg=sys.argv[1]
    visu_flg=int(sys.argv[2])
#
    c=0
    nums=[]; flns=[]
    while True:
        try:
            nums.append(int(sys.argv[c+3]))
            flns.append(sys.argv[c+4])
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
    flg=0
    for file in os.listdir('./'):
        filename = os.fsdecode(file)
        if filename.endswith(".vtp"):
            flg=1
            break
#
    if flg == 1:
        log.info('Please (re)move all vtp files from current directory (with e.g. make clean).')
        log.info('Exiting.')
        sys.exit(1)
    
#
    for i in range(nobj):
#
        log.info('='*60)
        log.info('Pack %2d of %10s: '%(nums[i],flns[i]))
        log.info('-'*60)
#
#       make the object from the input file
#
        obj=init(i,flns[i],log)
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
    if 'sox' in meth_flg:
        app=appdata(opt_1_x,n,nums,maps,vtps_0,c_l,c_a,c_r,2,0,1)
    elif 'six' in meth_flg:
        app=appdata(opt_0_x,n,nums,maps,vtps_0,c_l,c_a,c_r,1,0,1)
    else:
        app=appdata(opt_1_x,n,nums,maps,vtps_0,c_l,c_a,c_r,0,0,1)
#
    if visu_flg:
        vis=rndr(app)
    else:
        vis=None
#
    if meth_flg == 'objall':
#
#       dual annealing full collisions (based on objects) continuous rotations
#
        simu_args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,0,0)
        back_args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,nums,vtps,vtcs,0,0,log,vis)
        res=dual_annealing(simu_obp_co,args=simu_args,bounds=opt_1_bds,seed=1,maxfun=1000000,\
            callback=partial(back_da_co,args=back_args),no_local_search=True)
#
    elif meth_flg == 'objsix':
#
#       dual annealing full collisions (based on objects) 6 rotations
#
        simu_args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,1,0)
        back_args=(n,cols,tfms,vtps,maps,c_l,c_r,c_a,c_v_0,nums,vtps,vtcs,1,0,log,vis)
        res=dual_annealing(simu_obp_co,args=simu_args,bounds=opt_0_bds,seed=1,maxfun=1000000,\
            callback=partial(back_da_co,args=back_args))
#
    elif meth_flg == 'boxsix':
#
#       dual annealing axis aligned bounding box based collisions with 6 rotations
#
        simu_args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,1,0)
        back_args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,nums,vtps,vtcs,1,0,log,vis)
        res=dual_annealing(simu_obp,args=simu_args,bounds=opt_0_bds,seed=1,maxfun=1000000,\
            callback=partial(back_da,args=back_args))
#
    elif meth_flg == 'soxsix':
#
#       dual annealing axis aligned bounding box based collisions with 6 rotations
#       and stretching of object. add.
#
        simu_args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,2,0)
        back_args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,nums,vtps,vtcs,2,0,log,vis)
        res=dual_annealing(simu_obp,args=simu_args,bounds=opt_1_bds,seed=0,\
            callback=partial(back_da,args=back_args),no_local_search=True,\
            maxiter=int(1e6),maxfun=int(10e6))
#
    elif meth_flg == 'boxall':
#
#       dual annealing axis aligned bounding box based collisions with all rotations
#
        simu_args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,0,0)
        back_args=(n,pnts,maps,c_l,c_a,c_r,c_v_0,nums,vtps,vtcs,0,0,log,vis)
        res=dual_annealing(simu_obp,args=simu_args,bounds=opt_1_bds,seed=1,maxfun=1000000,\
            callback=partial(back_da,args=back_args))
#
    else:
        log.info('error')
#
    log.info('='*60)
    log.info('Scipy Output')
    log.info('-'*60)
#
    log.info(res)
#
    for i in range(n):
# 
        if 'sox' in meth_flg:
            tfm=tfmx(res.x,i,c_l,c_a,c_r,None,2,0)
            tmp=tran(vtps_0[maps[i]],tfm)
            woutfle(tmp,'build',-i-1)
        elif 'six' in meth_flg:
            tfm=tfmx(res.x,i,c_l,c_a,c_r,None,1,0)
            tmp=tran(vtps_0[maps[i]],tfm)
            woutfle(tmp,'build',-i-1)
        else:
            tfm=tfmx(res.x,i,c_l,c_a,c_r,None,0,0)
            tmp=tran(vtps_0[maps[i]],tfm)
            woutfle(tmp,'build',-i-1)
#
        tmp=[0.]*16
        tfm.GetMatrix().DeepCopy(tmp,tfm.GetMatrix())
        outs_0[i].extend(tmp)
#
    if 'sox' in meth_flg:
        app=appdata(res.x,n,nums,maps,vtps_0,c_l,c_a,c_r,2,0,1)
        woutfle(app.GetOutput(),'build',0)
        app=appdata(res.x,n,nums,maps,vtps,c_l,c_a,c_r,2,0,1)
        woutfle(app.GetOutput(),'objec',0)
        app=appdata(res.x,n,nums,maps,vtcs,c_l,c_a,c_r,2,0,1)
        woutfle(app.GetOutput(),'cubes',0)
    elif 'six' in meth_flg:
        app=appdata(res.x,n,nums,maps,vtps_0,c_l,c_a,c_r,1,0,1)
        woutfle(app.GetOutput(),'build',0)
        app=appdata(res.x,n,nums,maps,vtps,c_l,c_a,c_r,1,0,1)
        woutfle(app.GetOutput(),'objec',0)
        app=appdata(res.x,n,nums,maps,vtcs,c_l,c_a,c_r,1,0,1)
        woutfle(app.GetOutput(),'cubes',0)
    else:
        app=appdata(res.x,n,nums,maps,vtps_0,c_l,c_a,c_r,0,0,1)
        woutfle(app.GetOutput(),'build',0)
        app=appdata(res.x,n,nums,maps,vtps,c_l,c_a,c_r,0,0,1)
        woutfle(app.GetOutput(),'objec',0)
        app=appdata(res.x,n,nums,maps,vtcs,c_l,c_a,c_r,0,0,1)
        woutfle(app.GetOutput(),'cubes',0)
#
    with open('transforms.dat', 'w') as file:
        file.writelines('\t'.join(str(j) for j in i) + '\n' for i in outs_0)
#
    log.info('-'*60)
    log.info('Result written to build.vtp and transforms.dat')
    log.info('Load build.vtp in Paraview or with: python load.py build.vtp')
    log.info('='*60)
#
