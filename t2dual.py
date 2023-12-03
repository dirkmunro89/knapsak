#
import numpy as np
from scipy.optimize import minimize
#
def t2d(n,m,x_k,x_d_k,d_l,d_u,g,dg,ddg,c_s):
#
#   ddL=np.maximum(np.absolute(c_x[0] + np.dot(x_d_k,c_x[1:])),1e-12)
#
    bds=[[-1e1*(1-c_s[i]),1e1] for i in range(m)]; tup_bds=tuple(bds)
    sol=minimize(qp_dual,x_d_k.copy(),args=(n,m,x_k,g,dg,d_l,d_u,ddg), \
        jac=None,method='L-BFGS-B',bounds=tup_bds, \
        options={'disp':False,'gtol':1e-9,'ftol':1e-12,'maxls':1000})
#
    if sol.status != 0 or sol.success != True : 
        print('Warning; subproblem')
        print(sol)
        stop
#   print(sol.status)
#   print(sol)
#
    d=sol.x
#
    x=x_dual(d, n, m, x_k, g, dg, d_l, d_u, ddg)
#
    return x,d
#
# QP: x in terms of dual variables 
#
def x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddg):
#
    tmp=(dg[0]+np.dot(x_d,dg[1:]))
#
    tmp1=np.dot(np.linalg.inv(ddg),tmp)
#
    return np.maximum(np.minimum(x_k - tmp1, dx_u),dx_l)
#
# QP: Dual function value
#
def qp_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddg):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddg)
#
    W=g[0]+np.dot(dg[0],x-x_k)+1./2.*np.dot(np.dot(ddg,(x-x_k)).T,x-x_k)+np.dot(x_d,(g[1:]+np.dot(dg[1:],(x-x_k))))
#
    return -W
#
# QP: Dual gradient
#
def dqp_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddg):
#
    x=x_dual(x_d, n, m, x_k, g, dg, dx_l, dx_u, ddg)
#
    dW=g[1:]+np.dot(dg[1:],(x-x_k)) 
#
    return -dW
#

