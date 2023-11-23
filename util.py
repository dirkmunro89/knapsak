#
import vtk
import numpy as np
from scipy.spatial.transform import Rotation as R
#
def tfmx(x,c_l,c_a,c_r,int_flg):
#
    tfm=vtk.vtkTransform()
    tfm.PostMultiply()
#
    c=0
#
    if int_flg:
#
        tmp=abs(x[c*4])%7 - 3.5
#
        if tmp >= 0-3.5 and tmp < 1-3.5:
            r=R.from_matrix(c_r[0].T).as_rotvec()
        elif tmp >= 1-3.5 and tmp < 2-3.5:
            r=R.from_matrix(c_r[1].T).as_rotvec()
        elif tmp >= 2-3.5 and tmp < 3-3.5:
            r=R.from_matrix(c_r[2].T).as_rotvec()
        elif tmp >= 3-3.5 and tmp < 4-3.5:
            r=R.from_matrix(c_r[3].T).as_rotvec()
        elif tmp >= 4-3.5 and tmp < 5-3.5:
            r=R.from_matrix(c_r[4].T).as_rotvec()
        elif tmp >= 5-3.5 and tmp < 6-3.5:
            r=R.from_matrix(c_r[5].T).as_rotvec()
        elif tmp >= 6-3.5 and tmp < 7-3.5:
            r=R.from_matrix(c_r[6].T).as_rotvec()
        else:
            print(x[i*4],tmp)
            print('error util')
            exit()
#
        tmp=max(np.linalg.norm(r),1e-9)
        tfm.RotateWXYZ(np.rad2deg(tmp),r[0]/tmp,r[1]/tmp,r[2]/tmp)
        tfm.Translate(c_l[0]*x[c*4+1], c_l[1]*x[c*4+2], c_l[2]*x[c*4+3])
#
    else:
#
        tfm.RotateWXYZ(c_a*x[c*7], x[c*7+1], x[c*7+2], x[c*7+3])
        tfm.Translate(c_l[0]*x[c*7+4], c_l[1]*x[c*7+5], c_l[2]*x[c*7+6])
#
    tfm.Update()
#
    return tfm
#
def appdata(x,n,nums,maps,vtis,c_l,c_a,c_r,int_flg,str_flg,col_flg):
#
    t = np.linspace(-510, 510, n)                                              
    rgbs=np.round(np.clip(np.stack([-t, 510-np.abs(t), t], axis=1), 0, 255)).astype(np.uint8)
#
    c=0
    app=vtk.vtkAppendDataSets()
    app.SetOutputDataSetType(0)
    for i in range(len(nums)):
#
        if str_flg:
           red=vtk.vtkXMLPolyDataReader()
           red.ReadFromInputStringOn()
           red.SetInputString(stis[i])
           red.Update()
           vtp=red.GetOutput()
        else:
            vtp = vtis[i]
#
        for j in range(nums[i]):
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
                    print(x[c*4],tmp)
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
#
            if col_flg:
                color=vtk.vtkUnsignedCharArray() 
                color.SetName("Colors") 
                color.SetNumberOfComponents(3) 
                color.SetNumberOfTuples(tmp.GetNumberOfCells())
                for j in range(tmp.GetNumberOfCells()):
                    color_tup=list(rgbs[c])
                    color.SetTuple(j,color_tup)
                tmp.GetCellData().SetScalars(color)
#
            app.AddInputData(tmp)
#
            c=c+1
#
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
def woutfle(vtp,fln,k):
#
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(vtp)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToNone()
    if k < 0:
        writer.SetFileName(fln+'_%s.vtp'%(chr(ord('`')+(-k))))
    elif k == 0:
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
