#
import vtk
import numpy as np
from util import tran
#
def back_ga_na(xk,convergence,args):
#
    n=args[0]
    objs=args[1]
    c_l=args[2]
    c_a=args[3]
    c_v=args[4]
    [f,c]=simu_ga_na(xk,n,objs,c_l,c_a,c_v,1)
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
def simu_ga_na(x,n,string,c_l,c_a,c_v,flg):
#
    tfms=[]
    bnds=[]
    objs=[]
    apps=[]
#
    tfm_0 = vtk.vtkTransform()
    tfm_0.Translate(0., 0., 0.)
    tfm_0.Update()
    for i in range(n):
#
#   `read' in vtp objects from file strings
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(string[i])
        red.Update()
        obj = red.GetOutput()
#
#   move them
#
        tfm = vtk.vtkTransform()
        tfm.Translate(c_l[0]*x[i*7+0], c_l[1]*x[i*7+1], c_l[2]*x[i*7+2])
        tfm.RotateWXYZ(c_a*x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
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
        if i == 0:
            bds=list(bnds[i][:])
        else:
            for j in range(6):
                if j%2 == 0:
                    bds[j] = min(bds[j],bnds[i][j])
                else:
                    bds[j] = max(bds[j],bnds[i][j])
#
#   make appended data sets so that we have n-1 colision checks
#
    c=0
    for i in range(n-1):
        for j in range(i+1,n):
            coli = vtk.vtkCollisionDetectionFilter()
            coli.SetCollisionModeToAllContacts()
            coli.SetInputData(0, objs[i])
            coli.SetTransform(0, tfm_0)
            coli.SetInputData(1, objs[j])
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
