#
import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R
#
def appd3(x,n,nums,maps,stis,c_l,c_r):
#
    c=0
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for n in range(len(nums)):
#
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(stis[n])
        red.Update()
        vtp = red.GetOutput()
#
        for j in range(nums[n]):
#
            tfm=vtk.vtkTransform()
            tfm.Translate(c_l[0]*x[c*4+0], c_l[1]*x[c*4+1], c_l[2]*x[c*4+2])
#
            if x[c*4+3] >= 0-3.5 and x[c*4+3] < 1-3.5:
                r=R.from_matrix(c_r[0].T).as_rotvec()
#               tfm.RotateWXYZ(0, 1, 0, 0)
            elif x[c*4+3] >= 1-3.5 and x[c*4+3] < 2-3.5:
                r=R.from_matrix(c_r[1].T).as_rotvec()
#               tfm.RotateWXYZ(120, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3))
            elif x[c*4+3] >= 2-3.5 and x[c*4+3] < 3-3.5:
                r=R.from_matrix(c_r[2].T).as_rotvec()
#               tfm.RotateWXYZ(90, 1, 0, 0)
            elif x[c*4+3] >= 3-3.5 and x[c*4+3] < 4-3.5:
                r=R.from_matrix(c_r[3].T).as_rotvec()
#               tfm.RotateWXYZ(90, 0, 1, 0)
            elif x[c*4+3] >= 4-3.5 and x[c*4+3] < 5-3.5:
                r=R.from_matrix(c_r[4].T).as_rotvec()
#               tfm.RotateWXYZ(90, 0, 0, 1)
            elif x[c*4+3] >= 5-3.5 and x[c*4+3] < 6-3.5:
                r=R.from_matrix(c_r[5].T).as_rotvec()
#               tfm.RotateWXYZ(120, -1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3))
            elif x[c*4+3] >= 6-3.5 and x[c*4+3] < 7-3.5:
                r=R.from_matrix(c_r[6].T).as_rotvec()
#               tfm.RotateWXYZ(0, 1, 0, 0)
            else:
                print(x[i*4+3])
                print('error')
                exit()
#
            tmp=np.linalg.norm(r)
            tfm.RotateWXYZ(np.rad2deg(tmp), r[0]/max(tmp,1e-9),r[1]/max(tmp,1e-9),r[2]/max(tmp,1e-9))
#
#           if x[c*4+3] == 0:
#               tfm.RotateWXYZ(0, 1, 0, 0)
#           elif x[c*4+3] == 1:
#               tfm.RotateWXYZ(90, 1, 0, 0)
#           elif x[c*4+3] == 2:
#               tfm.RotateWXYZ(90, 0, 1, 0)
#           elif x[c*4+3] == 3:
#               tfm.RotateWXYZ(90, 0, 0, 1)
#           elif x[c*4+3] == 4:
#               tfm.RotateWXYZ(120, 1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3))
#           elif x[c*4+3] == 5:
#               tfm.RotateWXYZ(120, -1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3))
            tfm.Update()
            tmp=tran(vtp,tfm)
            app.AddInputData(tmp)
            c=c+1
    app.Update()
#
    return app
#
def appd2(x,nobj,objs_vtp,objs_num,c_l,c_a):
#
    c=0
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(nobj):
        obj = objs_vtp[i]
        for j in range(objs_num[i]):
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
def appd(x,nobj,objs_str,objs_num,c_l,c_a):
#
    c=0
    app = vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(nobj):
        red = vtk.vtkXMLPolyDataReader()
        red.ReadFromInputStringOn()
        red.SetInputString(objs_str[i])
        red.Update()
        obj = red.GetOutput()
        for j in range(objs_num[i]):
            tmp = x[7*c:7*c+7]
            [tmp,_,_] = move(obj,tmp[:3],tmp[3:7],c_l,c_a)
            app.AddInputData(tmp)
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
        writer.SetFileName(fln+'_all.vtp')
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
