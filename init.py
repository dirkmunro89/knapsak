#
import vtk
import numpy as np
from vtk.util import numpy_support
from scipy.spatial.transform import Rotation as R
#
from util import woutstr
#
#   Matlab-like struct
#
class Object():
    def __init__(self,**kwargs):
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
    def SetAttr(self,lab,val):
        self.__dict__[lab] = val
#
#   init part data structures
#
def init(i,fln,c_e,log):
#
    if 1 == 1:
#
        obj=Object()
        obj.idn=i
#
#       read the stl into polydata
#
        flt=vtk.vtkSTLReader()
        flt.SetFileName(fln)
        flt.Update()
#
        obj.vtp_0 = flt.GetOutput()
        obj.stp_0 = woutstr(obj.vtp_0)
#
#       get properties
#
        flt=vtk.vtkCenterOfMass()
        flt.SetInputData(obj.vtp_0)
        flt.SetUseScalarsAsWeights(False)
        flt.Update()
#
        obj.cen_0=flt.GetCenter()
#
        prp=vtk.vtkMassProperties()
        prp.SetInputData(obj.vtp_0)
        prp.Update() 
#
        obj.vol_0=prp.GetVolume()
        obj.srf_0=prp.GetSurfaceArea()
#
        log.info('-points   : %14d'%obj.vtp_0.GetNumberOfPoints())
        log.info('-cells    : %14d'%obj.vtp_0.GetNumberOfCells())
        log.info('-centroid : %14.3e'%(obj.cen_0[0]))
        log.info('          : %14.3e'%(obj.cen_0[1]))
        log.info('          : %14.3e'%(obj.cen_0[2]))
        log.info("-area     : %14.3e"%obj.srf_0)
        log.info("-volume   : %14.3e"%obj.vol_0)
#
#       clean it
#
        flt=vtk.vtkCleanPolyData()
        flt.SetInputData(obj.vtp_0)
        flt.SetTolerance(1e-4) # fraction of BB diagonal
        flt.Update()
#
        obj.vtp=flt.GetOutput()
#
#       only triangles
#
        flt=vtk.vtkTriangleFilter()
        flt.SetInputData(obj.vtp)
        flt.SetPassLines(False)
        flt.SetPassVerts(False)
        flt.Update()
#
        obj.vtp=flt.GetOutput()
#
        log.info('-'*60)
        log.info('Cleaned and decimated: ')
        log.info('-'*60)
#
        if obj.vtp.GetNumberOfCells() > c_e:
            flt=vtk.vtkQuadricDecimation()
            flt.SetInputData(obj.vtp)
            flt.SetTargetReduction((obj.vtp.GetNumberOfCells()-c_e)/obj.vtp.GetNumberOfCells())
            flt.SetVolumePreservation(True)
            flt.Update()
            obj.vtp=flt.GetOutput()
#
#       get new properties (cleaned / updated object)
#
        flt=vtk.vtkCenterOfMass()
        flt.SetInputData(obj.vtp)
        flt.SetUseScalarsAsWeights(False)
        flt.Update()
#
        obj.cen=flt.GetCenter()
#
        prp = vtk.vtkMassProperties()
        prp.SetInputData(obj.vtp)
        prp.Update()
#
        obj.vol=prp.GetVolume()
        obj.srf=prp.GetSurfaceArea()
#
        log.info('-points   : %14d'%obj.vtp.GetNumberOfPoints())
        log.info('-cells    : %14d'%obj.vtp.GetNumberOfCells())
        log.info('-centroid : %14.3e'%(obj.cen[0]))
        log.info('          : %14.3e'%(obj.cen[1]))
        log.info('          : %14.3e'%(obj.cen[2]))
        log.info("-area     : %14.3e"%obj.srf)
        log.info("-volume   : %14.3e"%obj.vol)
#
#       transform updated object so that its centroid is 0,0,0
#
        tfm_0=vtk.vtkTransform()
        tfm_0.Translate(-obj.cen[0], -obj.cen[1], -obj.cen[2])
        tfm_0.Update()
        flt=vtk.vtkTransformPolyDataFilter()
        flt.SetInputData(obj.vtp)
        flt.SetTransform(tfm_0)
        flt.Update()
#
        obj.vtp=flt.GetOutput()
        obj.stp=woutstr(obj.vtp)
#
#       get axis aligned bounds and bounding box volume
#
        obj.bds=obj.vtp.GetBounds()
        obj.bbv=(obj.bds[1]-obj.bds[0])*(obj.bds[3]-obj.bds[2])*(obj.bds[5]-obj.bds[4])
#
        log.info("-AAB box  : %14.3e"%(obj.bds[1]-obj.bds[0]))
        log.info("          : %14.3e"%(obj.bds[3]-obj.bds[2]))
        log.info("          : %14.3e"%(obj.bds[5]-obj.bds[4]))
        log.info("-with vol.: %14.3e"%obj.bbv)
#
#       make cube source
#
        src=vtk.vtkCubeSource()
        src.SetBounds(obj.bds)
        src.Update()
#
        obj.vtc=src.GetOutput()
#
#       clean it
#
        flt=vtk.vtkCleanPolyData()
        flt.SetInputData(obj.vtc)
        flt.SetTolerance(1e-4) # fraction of BB diagonal
        flt.Update()
#
        obj.vtc=flt.GetOutput()
#
#       triangles
#
        flt=vtk.vtkTriangleFilter()
        flt.SetInputData(obj.vtc)
        flt.Update()
#
        obj.vtc=flt.GetOutput()
        obj.stc=woutstr(obj.vtc)
#
#       get bounding box points
#
        obj.pts=numpy_support.vtk_to_numpy(obj.vtc.GetPoints().GetData())
#
        return obj
#
def pretfms():
#
    c_r=[]
    r=R.from_rotvec(0 * np.array([1.,1.,1.])).as_matrix().T 
    c_r.append(r)
    r=R.from_rotvec(2*np.pi/3 * np.array([1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(np.pi/2 * np.array([1,0,0])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(np.pi/2 * np.array([0,1,0])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(np.pi/2 * np.array([0,0,1])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(2*np.pi/3 * np.array([-1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)])).as_matrix().T
    c_r.append(r)
    r=R.from_rotvec(0 * np.array([1.,1.,1.])).as_matrix().T 
    c_r.append(r)
#
    return c_r
#
