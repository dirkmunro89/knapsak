#
import os
import sys
import vtk
import time
import cvxopt
import cplex
from cvxopt import matrix,solvers,spdiag
from scipy import sparse
import numpy as np
import logging as log
from functools import partial
from vtk.util import numpy_support
from scipy.optimize import minimize, dual_annealing
from scipy.spatial.transform import Rotation as R
#
from rndr import rndr
#
from init import init, pretfms6, pretfms24
from simu_obp import simu_obp, back_da
from simu_obp_co import simu_obp_co, back_da_co
from simu_obp_pt import simu_obp_pt, back_da_pt
from simu_obp_dt import simu_obp_dt, back_da_dt
#
from t2dual import t2d
#
from util import tfmx, tran, appdata, woutfle
#
def hess(xt,xk,col,pnts,nuts,c_a,c_l):
#
    g=np.zeros(len(xt))
    dposdt=np.zeros((len(xt),3))
#
    pos0=np.dot(xt[nuts[0]:nuts[1]],pnts[col[0]])+c_l*xk[7*col[0]+4:7*col[0]+7]
    pos1=np.dot(xt[nuts[1]:nuts[2]],pnts[col[1]])+c_l*xk[7*col[1]+4:7*col[1]+7]
#
    dposdt[nuts[0]:nuts[1]]=pnts[col[0]]#np.dot(pnt,rot)
    dposdt[nuts[1]:nuts[2]]=pnts[col[1]]#np.dot(pnt,rot)
#
    dis=pos0-pos1
#
    ddf=np.zeros((3,len(xt)))
#
#
    ddf=np.zeros((len(xt),len(xt)))
#
#   apparently only lower triangular is referenced. seems to work
#
    ddf[nuts[0]:nuts[1],nuts[0]:nuts[1]]=2*np.dot(dposdt[nuts[0]:nuts[1]],dposdt[nuts[0]:nuts[1]].T)
    ddf[nuts[1]:nuts[2],nuts[1]:nuts[2]]=2*np.dot(dposdt[nuts[1]:nuts[2]],dposdt[nuts[1]:nuts[2]].T)
    ddf[nuts[1]:nuts[2],nuts[0]:nuts[1]]=-2*np.dot(dposdt[nuts[1]:nuts[2]],dposdt[nuts[0]:nuts[1]].T)
    ddf[nuts[0]:nuts[1],nuts[1]:nuts[2]]=ddf[nuts[1]:nuts[2],nuts[0]:nuts[1]].T
#-2*np.dot(dposdt[nuts[1]:nuts[2]],dposdt[nuts[0]:nuts[1]].T)
#
#   for i in range(nuts[0],nuts[1]):
#       for j in range(nuts[0],nuts[1]):
#           ddf[i][j]=2.*np.dot(dposdt[i],dposdt[j])
#       for j in range(nuts[1],nuts[2]):
#           ddf[i][j]=-2.*np.dot(dposdt[i],dposdt[j])
#   for i in range(nuts[1],nuts[2]):
#       for j in range(nuts[0],nuts[1]):
#           ddf[i][j]=-2.*np.dot(dposdt[i],dposdt[j])
#       for j in range(nuts[1],nuts[2]):
#           ddf[i][j]=2.*np.dot(dposdt[i],dposdt[j])
#
    ddf_lst=[]
    for i in range(nt):
        ddf_lst_tmp1=[]
        ddf_lst_tmp2=[]
        for j in range(nt):
            ddf_lst_tmp1.append(j)
            ddf_lst_tmp2.append(ddf[i][j])
        ddf_lst.append([ddf_lst_tmp1,ddf_lst_tmp2])
#
    return ddf_lst
#
def grad(xt,xk,col,pnts,nuts,c_a,c_l):
#
    g=np.zeros(len(xt))
    dposdt=np.zeros((len(xt),3))
#
    pos0=np.dot(xt[nuts[0]:nuts[1]],pnts[col[0]])+c_l*xk[7*col[0]+4:7*col[0]+7]
    pos1=np.dot(xt[nuts[1]:nuts[2]],pnts[col[1]])+c_l*xk[7*col[1]+4:7*col[1]+7]
#
    dposdt[nuts[0]:nuts[1]]=pnts[col[0]]#np.dot(pnt,rot)
    dposdt[nuts[1]:nuts[2]]=pnts[col[1]]#np.dot(pnt,rot)
#
    dis=pos0-pos1
#
    df=np.zeros((3,len(xt)))
#
    df[0][nuts[0]:nuts[1]]=2.*np.dot(dposdt[nuts[0]:nuts[1]],dis)#/c_l[0]**2.
    df[0][nuts[1]:nuts[2]]=-2.*np.dot(dposdt[nuts[1]:nuts[2]],dis)#/c_l[0]**2.
#
    df[1][nuts[0]:nuts[1]]=np.ones(nuts[1]-nuts[0]) #+ np.ones(nuts[1]-nuts[0])#/(nuts[1]-nuts[0])
    df[2][nuts[1]:nuts[2]]=np.ones(nuts[2]-nuts[1]) #+ np.ones(nuts[2]-nuts[1])#/(nuts[2]-nuts[1])
#
#
    return df
#
def func(xt,xk,col,pnts,nuts,c_a,c_l):
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
#   pen=0.
#   pen=pen+(np.sum(xt[nuts[0]:nuts[0+1]])-1.)**2.
#   pen=pen+(np.sum(xt[nuts[1]:nuts[1+1]])-1.)**2.
#
    f = np.zeros(3)
    dis=pos0-pos1
    f[0]=np.dot(dis,dis.T)#/c_l[0]**2. #+ np.sum(xt)
    f[1]=(np.sum(xt[nuts[0]:nuts[0+1]])-1.)#/(nuts[1]-nuts[0])#**2.
    f[2]=(np.sum(xt[nuts[1]:nuts[1+1]])-1.)#/(nuts[2]-nuts[1])#**2.
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
    sys.argv=['prerot.py', 'objall', '0', '2', 'stl/3DBenchy.stl']
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
    xk=np.array([0. for i in range(7*n)])
    xk[4]=1.0/2.
    xk[6]=0.1
#   xk[0]=1.
#   xk[1]=0.
#   xk[2]=1.
#   xk[3]=1.
    xk=2.*np.random.rand(7*n)/4.-1./4.
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
    xt=np.zeros(nt)#/51
#
    g=np.zeros(3,dtype=float)
    dg=np.zeros((3,nt),dtype=float)
    c_x=np.zeros((3,nt),dtype=float)
    xd=np.zeros(2,dtype=float)
#
    gold=np.ones(3,dtype=float)*1e8
    golder=np.ones(3,dtype=float)*1e8
    xtold=xt.copy()
    xtolder=xt.copy()
    xdold=np.zeros(2,dtype=float)
    alpha=1.
#
#
#
    dx_l=np.ones(nt,dtype=float)
    dx_u=np.ones(nt,dtype=float)
#
    ddL=np.zeros(nt,dtype=float)
#
    dx_l=np.zeros(nt)-xt
    dx_u=np.ones(nt)-xt
#
    g=func(xt,xk,col,pnts_1,nuts,c_a,c_l)
    dg=grad(xt,xk,col,pnts_1,nuts,c_a,c_l)
    ddg=hess(xt,xk,col,pnts_1,nuts,c_a,c_l)
#
#   ddL=ddg[0]#np.maximum(ddg[0]+np.dot(x_d.transpose(),ddg[1:]),0e0)
    if 1 ==0:
        J=dg[0]/np.amax(c_l)**2.; ind = np.array(range(nt))
#       Q=sparse.csc_matrix((ddL, (ind, ind)), shape=(nt, nt))
        tmp=np.zeros((nt,nt),dtype=np.float64); np.fill_diagonal(tmp,1e0)
        A=np.append(dg[1:],-tmp,axis=0)
#       A=np.append(np.append(dg[1:],tmp,axis=0),-tmp,axis=0)
        u=-g[1:]; l=0.*np.ones(2,dtype=float)
#       l=np.append(l,dx_l); u=np.append(u,dx_u)
        h=-g[1:]
#       h=np.append(h,dx_u)
        h=np.append(h,-dx_l)
#
        P = matrix(ddg/np.amax(c_l)**2.,tc='d')
        q = matrix(J,tc='d')
        G = matrix(A,tc='d')
        h = matrix(h,tc='d')
#
        solvers.options['show_progress']=True
        sol=solvers.qp(P,q,G,h)
#
#   print(sol['x'])
#   print(sol['y'])
#
        xt[:]=np.array(sol['x']).flatten()
        xd[:]=np.array(sol['s'][:2]).flatten()
        g=func(xt,xk,col,pnts_1,nuts,c_a,c_l)
        print('g',g)
        print(sol['z'][:2])
    else:       
        prob=cplex.Cplex()
#       prob.set_results_stream(None)
#       prob.set_log_stream(None)
        prob.variables.add(obj=dg[0][:],lb=dx_l)#,ub=dx_u)
        ind = range(nt)#[i for i in range(n)]
        lin_expr=[]
        for j in range(2): lin_expr.append( cplex.SparsePair( ind = ind, val=dg[j+1] ))
        rhs=-g[1:]
        t1=time.time()
        prob.linear_constraints.add(lin_expr=lin_expr,rhs=rhs, senses=['L']*2)
        t2=time.time()
        print(t2-t1)
        prob.objective.set_quadratic(ddg)
        prob.parameters.threads=1
        prob.parameters.datacheck=0
        print(prob.parameters.threads)
        alg = prob.parameters.qpmethod.values
#       prob.parameters.qpmethod.set(alg.primal)
#       prob.parameters.qpmethod.set(alg.dual)
#       prob.parameters.qpmethod.set(alg.barrier)
        prob.solve()
        print(prob.solution.get_dual_values())
#
        xt[:]=prob.solution.get_values()
#
#   for i in range(10000):
#
#       if i>0:
#           golder[:]=gold
#           gold[:]=g
#
#       g=func(xt,xk,col,pnts_1,nuts,c_a,c_l)
#       dg=grad(xt,xk,col,pnts_1,nuts,c_a,c_l)
#       ddg=hess(xt,xk,col,pnts_1,nuts,c_a,c_l)
#       if i == 0:
#           gscl=max(g[0],1e-6)
#       g[0]=g[0]/gscl
#       dg[0]=dg[0]/gscl
#       ddg[0]=ddg[0]/gscl

#
#       if g[0] > gold[0]:# or np.amax(np.absolute(g[1:]))>1e-1:
#           xt[:]=xtold
#           xtold[:]=xtolder
#           xd[:]=xdold
#           alpha=alpha*2.
#           gold[:]=golder
#           g=func(xt,xk,col,pnts_1,nuts,c_a,c_l)
#           dg=grad(xt,xk,col,pnts_1,nuts,c_a,c_l)
#           ddg=hess(xt,xk,col,pnts_1,nuts,c_a,c_l)
#           g[0]=g[0]/gscl
#           dg[0]=dg[0]/gscl
#           ddg[0]=ddg[0]/gscl
#       else:
#           alpha=1.
#
#       print(i,g[0],xd,alpha)
#
#       c_s=np.ones(2)
#       mov=.1
#       x_u=np.ones(nt)
#       x_l=np.zeros(nt)
#       d_l = np.maximum(xt-mov*(x_u-x_l),x_l)
#       d_u = np.minimum(xt+mov*(x_u-x_l),x_u)
#       [xt_new,xd_new]=t2d(nt,2,xt,xd,d_l,d_u,g,dg,ddg*alpha,c_s)
#
#       xtolder[:]=xtold
#       xtold[:]=xt
#       xdold[:]=xd
#       xt[:]=xt_new
#       xd[:]=xd_new
#
#       if  alpha<1.1 and np.linalg.norm(xt-xtold)<1e-6:# and np.amax(np.absolute(g[1:]))<1e-2:
#           g=func(xt,xk,col,pnts_1,nuts,c_a,c_l)
#           print('break on', g)
#           break
#
#   bds=[[0.,1.] for i in range(nt)]; tup_bds=tuple(bds)
#   sol=minimize(func,xt,args=(xk,col,pnts_1,nuts,c_a,c_l),\
#       bounds=tup_bds,jac=grad,method='L-BFGS-B',options={'disp':True, 'maxiter':1000000})
#
#   print(sol.x)
#   xt=sol.x
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
#   print(sol.fun)
