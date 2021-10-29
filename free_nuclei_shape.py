import math
import trimesh

import numpy as np
from skimage import measure
from copy import deepcopy
# from stl import mesh
import matplotlib.tri as mtri

from analyze_stress_fibers import nucleus_reco_3d


def find_ellipsoid_semiaxis(scale_x, scale_y, scale_z, volume, ba_ratio, ca_ratio):
    """
    Finds ideal nucleus (ellipsoid) semi-axes based on formula: V = 4/3 πabc,
    where V is volume, a, b and c are axis_a * scale_x, axis_b * scale_y,
    axis_c * scale_z with correction to the pixel size of the original image.

    ---
        Parameters:
        - scale_x, scale_y, scale_z (double): pixel size of the original image
        - volume (double): volume of the original nucleus
        - ab_ratio (double): ratio of the semi-axis a to b of the ideal nucleus (ellipsoid)
        - ac_ratio (double): ratio of the semi-axis c to b of the ideal nucleus (ellipsoid)
    ---
        Returns:
        - axis_a, axis_b, axis_c (int): the semi-axes of the ideal nucleus (ellipsoid)
                                        in pixels of the original nucleus size
    """
    # V = 4/3 π * a * (ba_ratio * a) * (ca_ratio * a) -> a = (V / (4/3 * π * ba_ratio * ca_ratio))^1/3
    a = (volume / (4 / 3 * math.pi * ca_ratio * ba_ratio)) ** (1. / 3.)
    b = ba_ratio * a
    c = ca_ratio * a
    axis_a = int(a / scale_x)
    axis_b = int(b / scale_y)
    axis_c = int(c / scale_z)
    print(f"a = {a} micrometers")
    print(f"b = {b} micrometers")
    print(f"c = {c} micrometers")
    return axis_a, axis_b, axis_c


def draw_ideal_nucleus(axis_a, axis_b, axis_c):
    ideal_nucleus_3d = np.zeros((axis_a * 2 + 2, axis_b * 2 + 2, axis_c * 2 + 2), dtype=np.uint8)
    #ideal_nucleus_3d = np.zeros((axis_a * 2 + 2, axis_b * 2 + 2, axis_c * 2 + 2), dtype=mesh.Mesh.dtype)
    #ideal_nucleus_3d = deepcopy(A)

    # coordinates of a center
    x0, y0, z0 = ideal_nucleus_3d.shape[0] // 2, ideal_nucleus_3d.shape[1] // 2, ideal_nucleus_3d.shape[2] // 2

    for x in range(-axis_a, axis_a):
        for y in range(-axis_b, axis_b):
            for z in range(-axis_c, axis_c):
                # deb: The standard equation of an ellipsoid centered at the origin and aligned with the axes:
                # (x/a)^2 + (y/b)^2 + (z/c)^2 = 1 if deb is less than 1, the point is inside the ellipsoid.
                deb = (x / axis_a)**2 + (y / axis_b)**2 + (z / axis_c)**2
                if deb <= 1:
                    ideal_nucleus_3d[x + x0, y + y0, z + z0] = 1
    return ideal_nucleus_3d


def run_ideal_nucleus_creation(scale_x, scale_y, scale_z, volume, ba_ratio, ca_ratio):
    axis_a, axis_b, axis_c = find_ellipsoid_semiaxis(scale_x, scale_y, scale_z, volume, ba_ratio, ca_ratio)
    ideal_nucleus_3d = draw_ideal_nucleus(axis_a, axis_b, axis_c)
    verts, faces, normals, values = measure.marching_cubes(ideal_nucleus_3d, 0, step_size=1)
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('mesh_free_nuclei.stl')
    nucleus_reco_3d(ideal_nucleus_3d)
    print(f"The axis aligned box size of the free nuclei: {surf_mesh.extents}")


if __name__ == "__main__":
    scale_x, scale_y, scale_z = 0.04, 0.04, 0.17
    volume =773.9556000000006
    ba_ratio = 1/1#According presentation "1_20 Meeting" ratio A/B is 0.88692
    ca_ratio = 1/1   #According presentation "1_20 Meeting" ratio A/C is 1.204339
    run_ideal_nucleus_creation(scale_x, scale_y, scale_z, volume, ba_ratio, ca_ratio)

