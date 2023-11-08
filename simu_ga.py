#
import vtk
import numpy as np
from util import tran
from util import move
from util import appd
from util import woutfle
#
def back_sa(xk,fk,context,args):
#
    n=args[0]
    nobj=args[1]
    objs_str=args[2]
    objs_num=args[3]
    c_l=args[4]
    c_a=args[5]
    c_v=args[6]
    [f,c]=simu_ga(xk,n,nobj,objs_str,objs_num,c_l,c_a,c_v,1)
    print('%3d %14.3e %6d'%(context,f,c),flush=True)
#
#   app = vtk.vtkAppendDataSets()
#   app.SetOutputDataSetType(0)
#   for i in range(n):
#       red = vtk.vtkXMLPolyDataReader()
#       red.ReadFromInputStringOn()
#       red.SetInputString(objs[maps[i]])
#       red.Update()
#       obj = red.GetOutput()
#       tmp = xk[7*i:7*i+7]
#       [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
#       app.AddInputData(tmp)
#   app.Update()
#
    app=appd(xk,nobj,objs_str,objs_num,c_l,c_a)
    woutfle(app.GetOutput(),'see',-1)
#
    return False
#
def back_ga(xk,convergence,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu_ga(xk,n,objs,c_l,c_a,c_v,1)
    print('%7.3f %14.3e %6d'%(convergence,f,c),flush=True)
#
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(n):
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs[i])
        red.Update()
        obj = red.GetOutput()
        tmp = xk[7*i:7*i+7]
        [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
        app.AddInputData(tmp)
    app.Update()
    woutfle(app.GetOutput(),'see',-1)
#
    return False
#
def simu_ga(x,n,nobj,objs_str,objs_num,c_l,c_a,c_v,flg):
#
    tfms=[]
    bnds=[]
    objs=[]
    apps=[]
#
    c=0
    tfm_0 = vtk.vtkTransform()
    tfm_0.Translate(0., 0., 0.)
    tfm_0.Update()
    for i in range(nobj):
#
#   `read' in vtp objects from file strings
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs_str[i])
        red.Update()
        obj = red.GetOutput()
#
#   move them
#
        for j in range(objs_num[i]):
#
            tfm = vtk.vtkTransform()
            tfm.Translate(c_l[0]*x[c*7+0], c_l[1]*x[c*7+1], c_l[2]*x[c*7+2])
            tfm.RotateWXYZ(c_a*x[c*7+3], x[c*7+4], x[c*7+5], x[c*7+6])
            tfm.Update()
            vtp=tran(obj,tfm)
#
#   keep some stuff
#
            objs.append(vtp)
            bnds.append(vtp.GetBounds())
            tfms.append(tfm)
#
#   get total bounding box
#
            if c == 0:
                bds=list(bnds[c][:])
            else:
                for k in range(6):
                    if k%2 == 0:
                        bds[k] = min(bds[k],bnds[c][k])
                    else:
                        bds[k] = max(bds[k],bnds[c][k])
#
            c=c+1
#
#   make appended data sets so that we have n-1 colision checks (this seems to be faster)
#
    for i in range(n-1):
        app = vtk.vtkAppendDataSets()
        app.SetOutputDataSetType(0)
        for j in range(i+1,n):
            app.AddInputData(objs[j])
            app.Update()
        apps.append(app.GetOutput())
#
#   here is the n-1 colision checks
#
    c=0
    for i in range(n-1):
#
        coli = vtk.vtkCollisionDetectionFilter()
        coli.SetCollisionModeToAllContacts()
#       coli.SetCollisionModeToHalfContacts()
#       coli.SetCollisionModeToFirstContact()
        coli.SetInputData(0, objs[i])
        coli.SetTransform(0, tfm_0)
        coli.SetInputData(1, apps[i])
        coli.SetTransform(1, tfm_0)
        coli.Update()
        c=c+coli.GetNumberOfContacts()
#
    ext=100.
    f=bds[5]+ext
    f=(bds[1]-bds[0])*(bds[3]-bds[2])*(bds[5]-bds[4])/c_v
#
    f=f+c*c_v/n/309 #est. volume per triangle
#
#   if it has to fit in a box
#
    b=0.
    y=0.
    if bds[0]<-ext:
        b=b+(abs(bds[0])-ext)**1.
        y=y+1
    if bds[1]>ext:
        b=b+(abs(bds[1])-ext)**1.
        y=y+1
    if bds[2]<-ext:
        b=b+(abs(bds[2])-ext)**1.
        y=y+1
    if bds[3]>ext:
        b=b+(abs(bds[3])-ext)**1.
        y=y+1
    if bds[4]<-ext:
        b=b+(abs(bds[4]))**1.
        y=y+1
#
#   f=f+b*np.amax(c_l)
#
    if flg == 0:
        return f
    else:
        return f,c
#
