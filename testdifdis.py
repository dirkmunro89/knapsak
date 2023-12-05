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
from scipy.spatial.transform import Rotation as R
#
#
from rndr import rndr
#
from init import init, pretfms6, pretfms24
from simu_obp import simu_obp, back_da
from simu_obp_co import simu_obp_co, back_da_co
from simu_obp_pt import simu_obp_pt, back_da_pt
from simu_obp_dt import simu_obp_dt, back_da_dt
#
from util import tfmx, tran, appdata, woutfle
#
def grad(xt,xk,col,pnts,nuts,c_a,c_l,scl):
#
    g=np.zeros(len(xt))
    dposdt=np.zeros((len(xt),3))
#
    pos0=np.dot(xt[nuts[0]:nuts[1]],pnts[col[0]])+c_l*xk[7*col[0]+4:7*col[0]+7]
    pos1=np.dot(xt[nuts[1]:nuts[2]],pnts[col[1]])+c_l*xk[7*col[1]+4:7*col[1]+7]
#
#   dposdt[nuts[0]:nuts[1]]=pnts[col[0]]#np.dot(pnt,rot)
#   dposdt[nuts[1]:nuts[2]]=pnts[col[1]]#np.dot(pnt,rot)
#
    dis=pos0-pos1
    g[nuts[0]:nuts[1]]=2.*np.dot(pnts[col[0]],dis)/scl#np.amax(c_l)**2.
    g[nuts[1]:nuts[2]]=-2.*np.dot(pnts[col[1]],dis)/scl#np.amax(c_l)**2.
#
    g0=(np.sum(xt[nuts[0]:nuts[1]])-1.)#/(nuts[1]-nuts[0])
    g1=(np.sum(xt[nuts[1]:nuts[2]])-1.)#/(nuts[2]-nuts[1])
    if g0 > 0:
        g[nuts[0]:nuts[1]]=g[nuts[0]:nuts[1]]+2.*g0*1e2
    if g1 > 0:
        g[nuts[1]:nuts[2]]=g[nuts[1]:nuts[2]]+2.*g1*1e2
#
    return g
#
def func(xt,xk,col,pnts,nuts,c_a,c_l,scl):
#
    pos0=np.dot(xt[nuts[0]:nuts[1]],pnts[col[0]])+c_l*xk[7*col[0]+4:7*col[0]+7]
    pos1=np.dot(xt[nuts[1]:nuts[2]],pnts[col[1]])+c_l*xk[7*col[1]+4:7*col[1]+7]
#
#   for later
#   make each t ---> t_i*( t_t0 + t_t1 + t_t2 )**3.  
#   ---> I think it remains convexish. so, then, if triangle does not sum to 1, 
#   then the t does nothing--> 0. if it does, then the t counts.
#   else we do convex decompositions
#
    pen=0.
    g0=(np.sum(xt[nuts[0]:nuts[1]])-1.)#/(nuts[1]-nuts[0])
    g1=(np.sum(xt[nuts[1]:nuts[2]])-1.)#/(nuts[2]-nuts[1])
    if g0 > 0:
        pen=pen+g0**2.
    if g1 > 0:
        pen=pen+g1**2.
#
    dis=pos0-pos1
    f=np.dot(dis,dis.T)/scl+1e2*pen
#
    return f
#
if __name__ == "__main__":
#
#   parameters
#
    c_l=np.array([300.,300.,300.]) # for in box
    c_s=1.01
    c_a=np.pi 
    c_e=10000
#
#   get input arguments 
#   - number to be stacked and
#   - path to file
#
    t0=0#time.time()
    out='./out_%d/'%t0
    if not os.path.isdir(out):
        os.makedirs(out)
#   else:
#       print('error')
#       sys.exit(1)
    t0=time.time()
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
    sys.argv=['prerot.py', 'objall', '0', '2', 'stl/Cone.stl']
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
        obj=init(i,flns[i],c_e,c_s,log,0)
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
            col.SetGenerateScalars(0)
            col.SetInputData(0,objs[maps[i]].vtp)
            col.SetTransform(0,tfms[i])
            col.SetInputData(1,objs[maps[j]].vtp)
            col.SetTransform(1,tfms[j])
            col.Update()
            cols.append(col)
#
    log.info('='*60)
#
    cors = [obj.cor for obj in objs]
    pnts = [obj.pts for obj in objs]
    stps = [obj.stp for obj in objs]
    vtps = [obj.vtp for obj in objs]
    stcs = [obj.stc for obj in objs]
    vtcs = [obj.vtc for obj in objs]
    exts = [obj.ext for obj in objs]
    nrm_vecs = [obj.nrm_vec for obj in objs]
    nrm_pnts = [obj.nrm_pnt for obj in objs]
#
#   set up predefined transforms
#
    if '24r' in opt_str:
        c_r=pretfms24()
    elif 'six' in opt_str:
        c_r=pretfms6()
    else:
        c_r=[]
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
    log.info('%6s%15s'%('k','F_0 (opt)'))
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
#   if 'sox' in opt_str:
#       app=appdata(opt_1_x,n,nums,maps,vtps_0,c_l,c_a,c_r,2,0,1)
#   elif 'six' in opt_str:
#       app=appdata(opt_0_x,n,nums,maps,vtps_0,c_l,c_a,c_r,1,0,1)
#   elif '24r' in opt_str:
#       app=appdata(opt_0_x,n,nums,maps,vtps_0,c_l,c_a,c_r,24,0,1)
#   else:
#       app=appdata(opt_1_x,n,nums,maps,vtps_0,c_l,c_a,c_r,0,0,1)
#
    if vis_flg:
        vis=rndr(app)
    else:
        vis=None
#
    xk=np.array([0 for i in range(7*n)])
#   xk[4]=1.
#   xk[0]=1.
#   xk[1]=0.
#   xk[2]=1.
#   xk[3]=1.
    xk=2.*np.random.rand(7*n)/2.-1./2.
#
#
#
#   will always be only two parts, so maybe
#
    col=[0,1]
#
    nt=0
    nuts=[nt]
    nt=nt+len(pnts[maps[col[0]]])
    nuts.append(nt)
    nt=nt+len(pnts[maps[col[1]]])
    nuts.append(nt)
#
#   and update points once, then pass
#
    pnts_1=[]
    for i in range(n):
        pnt=pnts[maps[i]]
        tmp = np.array([xk[7*i+1],xk[7*i+2],xk[7*i+3]])
        if np.linalg.norm(tmp):
            tmp = tmp/np.linalg.norm(tmp)
        else:
            tmp=tmp*0.
        rot=R.from_rotvec((c_a*xk[7*i])*tmp).as_matrix().T
        pnts_1.append(np.dot(pnt,rot))
#
    xt=np.zeros(nt)#np.ones(nt)/nt*2.
#
    scl=np.linalg.norm(c_l*xk[7*col[0]+4:7*col[0]+7]-c_l*xk[7*col[1]+4:7*col[1]+7])**2.
#
    print('qp')
    t0=time.time()
    bds=[[0.,1.] for i in range(nt)]; tup_bds=tuple(bds)
    sol=minimize(func,xt,args=(xk,col,pnts_1,nuts,c_a,c_l,scl),\
        bounds=tup_bds,jac=grad,method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-12})#,\
#       options={'disp':False,'gtol':1e-12,'ftol':1e-12,'maxls':1000})
#       bounds=tup_bds,jac=grad,method='TNC',\
#   sol=minimize(func,sol.x,args=(xk,col,pnts_1,nuts,c_a,c_l),\
#       bounds=tup_bds,jac=grad,method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-12})#,\
#
    t1=time.time()
    xt=sol.x
    print('distance=',np.sqrt(sol.fun*scl))
#
    tees=[]
    for i in range(n):
        tees.append(xt[nuts[i]:nuts[i+1]])
#
    app=appdata(xk,n,nums,maps,vtps,tees,c_l,c_a,c_r,0,0,1)
    woutfle(out,app.GetOutput(),'objec',0)
#
    points=vtk.vtkPoints()
#
    ct=0
    pees=[]
    pen=0.
    for i in range(n):
#
#       update points and compute vec
        pnt=pnts[maps[i]]
        tmp = np.array([xk[7*i+1],xk[7*i+2],xk[7*i+3]])
        if np.linalg.norm(tmp):
            tmp = tmp/np.linalg.norm(tmp)
        else:
            tmp=tmp*0.
        rot=R.from_rotvec((c_a*xk[7*i])*tmp).as_matrix().T
        pos=np.dot(xt[ct:ct+len(pnt)],np.dot(pnt,rot))
#
        pen=pen+(np.sum(xt[ct:ct+len(pnt)])-1.)**2.
        print('sum',np.sum(xt[ct:ct+len(pnt)]))
        ct=ct+len(pnt)
#
        pee=pos+xk[7*i+4:7*i+7]*c_l
        points.InsertNextPoint(pee)
#
        pees.append(pee)
#
    line=vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(2)
    line.GetPointIds().SetId(0,0)
    line.GetPointIds().SetId(1,1)
    cell=vtk.vtkCellArray()
    cell.InsertNextCell(line)
    ply=vtk.vtkPolyData()
    ply.SetPoints(points)
    ply.SetLines(cell)
    woutfle(out,ply,'line',0)
#
#
#   check
#
    vtps_1=[]
    for i in range(n):
#
        tfmx(xk,i,c_l,c_a,c_r,tfms[i],99,0)
#
        vtp=tran(vtps[maps[i]],tfms[i]) # can maybe get this from col object
#
        vtps_1.append(vtp)

#
    flt0=vtk.vtkImplicitPolyDataDistance()
    flt1=vtk.vtkImplicitPolyDataDistance()
#
    flt0.SetInput(vtps_1[0])
    flt1.SetInput(vtps_1[1])
#
    pnt0=vtps_1[0].GetPoints()
    pnt1=vtps_1[1].GetPoints()
#
    min_sd0=1e8
    min_sd1=1e8
    min_p0=0.
    min_p1=0.
    for pid in range(pnt0.GetNumberOfPoints()):
        p = pnt0.GetPoint(pid)
        sd = flt1.EvaluateFunction(p)
        if sd < min_sd0:
            min_p0=p
        min_sd0=min(min_sd0,sd)
    for pid in range(pnt1.GetNumberOfPoints()):
        p = pnt1.GetPoint(pid)
        sd = flt0.EvaluateFunction(p)
        if sd < min_sd1:
            min_p1=p
        min_sd1=min(min_sd1,sd)
#
    points=vtk.vtkPoints()
    points.InsertNextPoint(min_p0)
    points.InsertNextPoint(min_p1)
    print(max(min_sd0,0.))
    print(max(min_sd1,0.))
    line=vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(2)
    line.GetPointIds().SetId(0,0)
    line.GetPointIds().SetId(1,1)
    cell=vtk.vtkCellArray()
    cell.InsertNextCell(line)
    ply=vtk.vtkPolyData()
    ply.SetPoints(points)
    ply.SetLines(cell)
    woutfle(out,ply,'line_sd',0)
#
    stop
#
