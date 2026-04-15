from __future__ import annotations

import numpy as np
import vtk

from .io_vtk import numpy_from_vtk_matrix, vtk_matrix_from_numpy


def apply_matrix_to_polydata(polydata: vtk.vtkPolyData, mat4: np.ndarray) -> vtk.vtkPolyData:
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix_from_numpy(mat4))

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputData(polydata)
    tf.SetTransform(transform)
    tf.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(tf.GetOutput())
    return out


def compose_matrices(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Compose transforms under column-vector convention:
      p_out = second @ first @ p_in
    """
    return second @ first


def vtk_transform_to_numpy_4x4(transform: vtk.vtkAbstractTransform) -> np.ndarray:
    if isinstance(transform, vtk.vtkLinearTransform):
        return numpy_from_vtk_matrix(transform.GetMatrix())
    raise TypeError("Only linear VTK transforms can be converted to 4x4 matrices.")
