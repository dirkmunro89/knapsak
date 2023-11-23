#
import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R
#
def appdata(x,n,nums,maps,vtis,c_l,c_a,c_r,int_flg,str_flg):
#
    c=0
    app=vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for n in range(len(nums)):
#
        if str_flg:
           red=vtk.vtkXMLPolyDataReader()
           red.ReadFromInputStringOn()
           red.SetInputString(stis[n])
           red.Update()
           vtp=red.GetOutput()
        else:
            vtp = vtis[n]
#
        for j in range(nums[n]):
#
            tfm=vtk.vtkTransform()
            tfm.PostMultiply()
#
            if int_flg:
#
#               derivative of modulo operator is 1 (in fixed point arithmetic) :)
                tmp=abs(x[c*4])%7 - 3.5
#
                if tmp >= 0-3.5 and tmp < 1-3.5:
                    r=R.from_matrix(c_r[0].T).as_rotvec()
#               tfm.RotateWXYZ(0, 1, 0, 0)
                elif tmp >= 1-3.5 and tmp < 2-3.5:
                    r=R.from_matrix(c_r[1].T).as_rotvec()
#               tfm.RotateWXYZ(120, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3))
                elif tmp >= 2-3.5 and tmp < 3-3.5:
                    r=R.from_matrix(c_r[2].T).as_rotvec()
#               tfm.RotateWXYZ(90, 1, 0, 0)
                elif tmp >= 3-3.5 and tmp < 4-3.5:
                    r=R.from_matrix(c_r[3].T).as_rotvec()
#               tfm.RotateWXYZ(90, 0, 1, 0)
                elif tmp >= 4-3.5 and tmp < 5-3.5:
                    r=R.from_matrix(c_r[4].T).as_rotvec()
#               tfm.RotateWXYZ(90, 0, 0, 1)
                elif tmp >= 5-3.5 and tmp < 6-3.5:
                    r=R.from_matrix(c_r[5].T).as_rotvec()
#               tfm.RotateWXYZ(120, -1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3))
                elif tmp >= 6-3.5 and tmp < 7-3.5:
                    r=R.from_matrix(c_r[6].T).as_rotvec()
#               tfm.RotateWXYZ(0, 1, 0, 0)
                else:
                    print(x[i*4],tmp)
                    print('error util')
                    exit()
                tmp=max(np.linalg.norm(r),1e-9)
                tfm.RotateWXYZ(np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
                tfm.Translate(c_l[0]*x[c*4+1], c_l[1]*x[c*4+2], c_l[2]*x[c*4+3])
            else:
                tfm.RotateWXYZ(c_a*x[c*7], x[c*7+1], x[c*7+2], x[c*7+3])
                tfm.Translate(c_l[0]*x[c*7+4], c_l[1]*x[c*7+5], c_l[2]*x[c*7+6])
#
            tfm.Update()
            tmp=tran(vtp,tfm)
            app.AddInputData(tmp)
#
            c=c+1
#
    app.Update()
#
    return app
#
def appd2(x,n,nums,maps,vtis,c_l,c_r):
#
    c=0
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(len(nums)):
#       obj=vtis[i]
        for j in range(nums[i]):
            tfm=vtk.vtkTransform()
            tfm.Translate(c_l[0]*x[c*4+0], c_l[1]*x[c*4+1], c_l[2]*x[c*4+2])
            if x[c*4+3] == 0:
                tfm.RotateWXYZ(0, 1, 0, 0)
            elif x[c*4+3] == 1:
                tfm.RotateWXYZ(90, 1, 0, 0)
            elif x[c*4+3] == 2:
                tfm.RotateWXYZ(90, 0, 1, 0)
            elif x[c*4+3] == 3:
                tfm.RotateWXYZ(90, 0, 0, 1)
            elif x[c*4+3] == 4:
                tfm.RotateWXYZ(120, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3))
            elif x[c*4+3] == 5:
                tfm.RotateWXYZ(120, -1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3))
            tfm.Update()
            tmp=tran(obj,tfm)
            app.AddInputData(tmp)
            c=c+1
    app.Update()
#
    return app
#
def appd(x,n,nums,maps,stis,c_l,c_a):
#
    c=0
    app=vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(len(nums)):
        red=vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(stis[i])
        red.Update()
        vtp=red.GetOutput()
        for j in range(nums[i]):
            tmp=x[7*c:7*c+7]
            tfm=vtk.vtkTransform()
            tfm.Translate(c_l[0]*tmp[0], c_l[1]*tmp[1], c_l[2]*tmp[2])
            tfm.RotateWXYZ(c_a*tmp[3], tmp[4], tmp[5], tmp[6])
            tfm.Update()
            flt=vtk.vtkTransformPolyDataFilter()
            flt.SetInputData(vtp)
            flt.SetTransform(tfm)
            flt.Update()
            tmp_vtp=flt.GetOutput()
#           [vtp,_,_] = move(vtp,tmp[:3],tmp[3:7],c_l,c_a)
            app.AddInputData(tmp_vtp)
            c=c+1
    app.Update()
#
    return app
#
def tran(vtp,tfm):
#
    tfm_flt =  vtk.vtkTransformPolyDataFilter()
    tfm_flt.SetInputData(vtp)
    tfm_flt.SetTransform(tfm)
    tfm_flt.Update()
    tmp = tfm_flt.GetOutput()
#
    return tmp
#
def move(vtp,trs,rot,c_l,c_a):
#
    tfm = vtk.vtkTransform()
    tfm.Translate(c_l[0]*trs[0], c_l[1]*trs[1], c_l[2]*trs[2])
    tfm.RotateWXYZ(c_a*rot[0], rot[1], rot[2], rot[3])
    tfm.Update()
    tfm_flt =  vtk.vtkTransformPolyDataFilter()
    tfm_flt.SetInputData(vtp)
    tfm_flt.SetTransform(tfm)
    tfm_flt.Update()
    vtp = tfm_flt.GetOutput()
#
    return [vtp,tfm,tfm_flt]
#
def woutfle(vtp,fln,k):
#
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtp)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    if k < 0:
        writer.SetFileName(fln+'.vtp')
    else:
        writer.SetFileName(fln+'_%d.vtp'%k)
    writer.Update()
#
def woutstr(vtp):
#
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtp)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
#   writer.SetCompressorTypeToZLib()
    writer.SetWriteToOutputString(True)
    writer.Update()
    return writer.GetOutputString()
#
