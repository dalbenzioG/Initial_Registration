from __future__ import annotations

import vtk


def clean_and_decimate(polydata: vtk.vtkPolyData, target_reduction: float = 0.0) -> vtk.vtkPolyData:
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(polydata)
    clean.Update()
    current = clean.GetOutput()

    if target_reduction > 0.0:
        dec = vtk.vtkDecimatePro()
        dec.SetInputData(current)
        dec.SetTargetReduction(target_reduction)
        dec.PreserveTopologyOn()
        dec.Update()
        current = dec.GetOutput()

    out = vtk.vtkPolyData()
    out.DeepCopy(current)
    return out
