from __future__ import annotations

import numpy as np
import nibabel as nib
import vtk
from skimage.measure import marching_cubes


def vtk_matrix_from_numpy(mat: np.ndarray) -> vtk.vtkMatrix4x4:
    vtk_mat = vtk.vtkMatrix4x4()
    for row in range(4):
        for col in range(4):
            vtk_mat.SetElement(row, col, float(mat[row, col]))
    return vtk_mat


def numpy_from_vtk_matrix(vtk_mat: vtk.vtkMatrix4x4) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    for row in range(4):
        for col in range(4):
            mat[row, col] = vtk_mat.GetElement(row, col)
    return mat


def apply_matrix_to_points(points_xyz: np.ndarray, mat4: np.ndarray) -> np.ndarray:
    """
    Apply homogeneous transform to Nx3 points.
    Convention: p_out_h = mat4 @ p_in_h (column-vector math).
    """
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([points_xyz, ones])
    out_h = (mat4 @ pts_h.T).T
    return out_h[:, :3]


def make_vtk_polydata(vertices: np.ndarray, faces: np.ndarray) -> vtk.vtkPolyData:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(vertices.shape[0])
    for idx, point in enumerate(vertices):
        vtk_points.SetPoint(idx, float(point[0]), float(point[1]), float(point[2]))

    vtk_cells = vtk.vtkCellArray()
    for tri in faces:
        cell = vtk.vtkTriangle()
        cell.GetPointIds().SetId(0, int(tri[0]))
        cell.GetPointIds().SetId(1, int(tri[1]))
        cell.GetPointIds().SetId(2, int(tri[2]))
        vtk_cells.InsertNextCell(cell)

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetPolys(vtk_cells)

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clean.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out


def load_binary_segmentation_nii_as_polydata(
    nii_path: str,
    label_value: int = 1,
    smoothing_iterations: int = 0,
) -> vtk.vtkPolyData:
    img = nib.load(nii_path)
    data = np.asanyarray(img.dataobj)
    affine = img.affine

    if data.ndim != 3:
        raise ValueError(f"Expected 3D segmentation, got {data.shape}")

    mask = data == label_value
    if not np.any(mask):
        unique_vals = np.unique(data)
        raise ValueError(
            f"Label {label_value} not found in {nii_path}. "
            f"Available labels sample: {unique_vals[:20]}"
        )

    verts_ijk, faces, _normals, _values = marching_cubes(mask.astype(np.float32), level=0.5)
    verts_world = apply_matrix_to_points(verts_ijk[:, [0, 1, 2]], affine)
    poly = make_vtk_polydata(verts_world, faces)

    if smoothing_iterations > 0:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputData(poly)
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        out = vtk.vtkPolyData()
        out.DeepCopy(smoother.GetOutput())
        poly = out

    return poly


def vtk_points_to_numpy(vtk_points: vtk.vtkPoints) -> np.ndarray:
    n_points = vtk_points.GetNumberOfPoints()
    points = np.empty((n_points, 3), dtype=np.float64)
    for i in range(n_points):
        points[i] = vtk_points.GetPoint(i)
    return points
