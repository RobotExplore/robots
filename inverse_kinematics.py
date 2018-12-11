import numpy as np
import sympy
from functools import reduce
from robot_kinematics import rotate, trans

AXIS = ('X', 'Y', 'Z')


def rotate_sym(axis, deg):
    axis = str(axis).upper()
    if axis not in AXIS:
        print(f"{axis} is unknown axis, should be one of {AXIS}")
        return
    rot_x = axis == 'X'
    rot_y = axis == 'Y'
    rot_z = axis == 'Z'
    rot_mat = sympy.Matrix([[(sympy.cos(deg), 1)[rot_x], (0, -sympy.sin(deg))[rot_z], (0, sympy.sin(deg))[rot_y], 0],
                            [(0, sympy.sin(deg))[rot_z], (sympy.cos(deg), 1)[rot_y], (0, -sympy.sin(deg))[rot_x], 0],
                            [(0, -sympy.sin(deg))[rot_y], (0, sympy.sin(deg))[rot_x], (sympy.cos(deg), 1)[rot_z], 0],
                            [0, 0, 0, 1]])
    # rot_mat = np.where(np.abs(rot_mat) < 1e-10, 0, rot_mat)  # get a small value when np.cos(np.pi/2)
    return rot_mat


def trans_sym(axis, dis):
    axis = str(axis).upper()
    if axis not in AXIS:
        print(f"{axis} is unknown axis, should be one of {AXIS}")
        return
    trans_mat = sympy.eye(4)
    trans_mat[AXIS.index(axis), 3] = dis
    return trans_mat

thea_1, thea_2, thea_3, thea_4, thea_5, thea_6 = sympy.symbols("j1, j2, j3, j4, j5, j6")

def fk_sym(joints):
    pass
DH = [[thea_1, 0.3991, 0, -np.pi / 2],
      [thea_2, 0, 0.448, 0],
      [thea_3, 0, 0.042, -np.pi / 2],
      [thea_4, 0.451, 0, np.pi / 2],
      [thea_5, 0, 0, -np.pi / 2],
      [thea_6, 0.082, 0, 0]]

# c1, s1 = sympy.symbols("c1, s1")
# A = sympy.Matrix([[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# B = np.dot(trans('z', 399.1), rotate('x', -np.pi / 2))
T60 = sympy.eye(4)
T_all = []
for i, (thea_i, d_i, l_i, a_i) in enumerate(DH, 1):
    temp = np.dot(np.dot(trans('z', d_i), trans('x', l_i)), rotate('x', a_i))
    T = rotate_sym('z', thea_i) * temp
    T60 = T60*T
    T_all.append(T)
    print(f"the {i}th joint: \n{T}")
print(f"T60: \n {T60}")

