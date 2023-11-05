#
import vtk
import numpy as np
from util import tran
#
def simu_ga(x,n,string,c_l,c_a,c_v,flg):
#
    tfms=[]
    bnds=[]
    objs=[]
    apps=[]
#
    tfm_0 = vtk.vtkTransform()
    tfm_0.Translate(0., 0., 0.)
    tfm_0.Update()
#
    for i in range(n):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(string[i])
        red.Update()
        obj = red.GetOutput()
#
        tfm = vtk.vtkTransform()
        tfm.Translate(c_l[0]*x[i*7+0], c_l[1]*x[i*7+1], c_l[2]*x[i*7+2])
        tfm.RotateWXYZ(c_a*x[i*7+3], x[i*7+4], x[i*7+5], x[i*7+6])
        tfm.Update()
#
        vtp=tran(obj,tfm)
        objs.append(vtp)
        bnds.append(vtp.GetBounds())
        tfms.append(tfm)
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
    for i in range(n-1):
        app = vtk.vtkAppendDataSets()
        app.SetOutputDataSetType(0)
        for j in range(i+1,n):
            app.AddInputData(objs[j])
            app.Update()
        apps.append(app.GetOutput())
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
