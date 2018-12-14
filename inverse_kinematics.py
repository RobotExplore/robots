# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv, pinv, solve, norm
import sympy
from functools import reduce
# from robot_kinematics import rotate, trans
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# %%
AXIS = ('X', 'Y', 'Z')
sympy.init_printing()


def rotate_sym(axis, deg):
    axis = str(axis).upper()
    if axis not in AXIS:
        print(f"{axis} is unknown axis, should be one of {AXIS}")
        return
    rot_x = axis == 'X'
    rot_y = axis == 'Y'
    rot_z = axis == 'Z'
    rot_mat = sympy.Matrix([[(sympy.cos(deg), 1)[rot_x], (0, -sympy.sin(deg))[rot_z], (0, sympy.sin(deg))[rot_y], 0],
                            [(0, sympy.sin(deg))[rot_z], (sympy.cos(deg), 1)
                             [rot_y], (0, -sympy.sin(deg))[rot_x], 0],
                            [(0, -sympy.sin(deg))[rot_y], (0, sympy.sin(deg))
                             [rot_x], (sympy.cos(deg), 1)[rot_z], 0],
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


def rotate_any_sym(f, a):
    va = 1 - sympy.cos(a)
    sin, cos = sympy.sin, sympy.cos
    T_rot = sympy.Matrix([[f[0]**2*va + cos(a), f[1]*f[0]*va - f[2]*sin(a), f[2]*f[0]*va + f[1]*sin(a), 0],
                          [f[0]*f[1]*va + f[2] *
                              sin(a), f[1]**2*va + cos(a), f[2]*f[1]*va - f[0]*sin(a), 0],
                          [f[0]*f[2]*va - f[1] *
                              sin(a), f[1]*f[2]*va + f[0]*sin(a), f[2]**2*va + cos(a), 0],
                          [0, 0, 0, 1]])
    return T_rot


def fk_sym(joints_sym):
    thea_1, thea_2, thea_3, thea_4, thea_5, thea_6 = joints_sym
    DH = [[thea_1, 0.3991, 0, -sympy.pi / 2],
          [thea_2, 0, 0.448, 0],
          [thea_3, 0, 0.042, -sympy.pi / 2],
          [thea_4, 0.451, 0, sympy.pi / 2],
          [thea_5, 0, 0, -sympy.pi / 2],
          [thea_6, 0.082, 0, 0]]
    T60 = sympy.eye(4)
    # T_all = []
    for i, (thea_i, d_i, l_i, a_i) in enumerate(DH, 1):
        T = rotate_sym('z', thea_i) * trans_sym('z', d_i) * \
            trans_sym('x', l_i) * rotate_sym('x', a_i)
        T60 = T60*T
        # T_all.append(T)
        # print(f"the {i}th joint: \n{T}")
    # print(f"T60: \n {T60}")
    return T60


def jacobian_sym(joints_sym):
    T60_expr = fk_sym(joints_sym).reshape(16, 1)
    ja = []
    for i in range(12):
        d = [sympy.diff(T60_expr[i], angle) for angle in joints_sym]
        ja.append(d)
    return sympy.Matrix(ja)


# transfer symbol expression to numpy functions to calculate faster
joints_sym = sympy.symbols("j1, j2, j3, j4, j5, j6")
fk_cal = sympy.lambdify(joints_sym, fk_sym(joints_sym), "numpy")
ja_cal = sympy.lambdify(joints_sym, jacobian_sym(joints_sym), "numpy")

# rotate around any axis
f_axis = sympy.symbols("f_x, f_y, f_z")
angle = sympy.symbols('a')
rotate_any = sympy.lambdify(
    (f_axis, angle), rotate_any_sym(f_axis, angle), "numpy")


def forward_kinematics(joints):
    T = fk_cal(joints[0], joints[1], joints[2],
               joints[3], joints[4], joints[5])
    return np.where(np.abs(T) < 1e-8, 0, T)


def jacobian(joints):
    return ja_cal(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])

# %%


def ik(T_tar, joints_ini):
    step = 0.5
    lam = 0.5
    joints_cu = joints_ini
    iteration = 0
    while True:
        T_cu = forward_kinematics(joints_cu)
        delta_p = T_tar - T_cu
        delta_p = delta_p.flatten()[:12]
        error = norm(delta_p)
        # if iteration % 10 == 0:
        #     print(f"Iteration {iteration}, error:{error}")
        if error < 1e-4 or iteration > 1e3:
            # print(f"Iteration {iteration}, error:{error}")
            return joints_cu
        ja = jacobian(joints_cu)
        # pseudo inverse
        delta_joints = np.dot(pinv(ja), delta_p)
        # damped least square
        # f = solve(ja.dot(ja.transpose())+lam**2*np.eye(12), delta_p)
        # delta_joints = np.dot(ja.transpose(), f)
        # inv_t = inv(ja.transpose().dot(ja) + lam**2*np.eye(6))
        # delta_joints = inv_t.dot(ja.transpose()).dot(delta_p)

        joints_cu = joints_cu + delta_joints*step
        iteration += 1


# %%
if __name__ == "__main__":
    joints_ini = [0, -np.pi/2, 0, 0, 0, np.pi]
    joints_ini = np.random.random(6)
    # joints_set = np.random.uniform(0, 2*np.pi, size=(100, 6))
    # end_positions = np.array([forward_kinemacitcs(joints)
    #                           for joints in joints_set])
    end_ini = forward_kinematics(joints_ini)
    p1 = np.array([0.04, 0.02, 0.34])
    p2 = np.array([0.08, 0.01, 0.07])
    f_axis = p2 - p1
    f_axis = f_axis / norm(f_axis)
    p12 = np.vstack((p1, p2))
    end_cu = end_ini.copy()
    end_cu[:3, 3] -= p1
    end_positions = end_ini.copy().reshape(1, 4, 4)
    for angle in np.linspace(0, np.pi*2, 100):
        end_next = np.dot(rotate_any(f_axis, angle), end_cu)
        end_next[:3, 3] += p1
        end_positions = np.vstack((end_positions, end_next.reshape(1, 4, 4)))
        
    end_next.shape
    # end_positions
    # joints_ini = joints_set[0]
    # %%
    joints_cu = joints_ini.copy()
    joints_rst = joints_ini.copy()
    for end in end_positions[1:]:
        jonints_cu = ik(end, joints_cu)
        joints_rst = np.vstack((joints_rst, jonints_cu))
    joints_rst.shape
    # %%
    # end_positions[0:2]
    # joints_rst.shape
    end_positions_pred = np.array([forward_kinematics(joints)
                                for joints in joints_rst])
    end_positions_pred.shape

    # %%
    # end_positions[-1] - end_positions_pred[-1]
    # %%
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(p12[:, 0], p12[:, 1], p12[:, 2], 'g-')
    ax.plot(end_positions[:, 0, 3],
            end_positions[:, 1, 3],
            end_positions[:, 2, 3], 'r-')
    ax.plot(end_positions_pred[:, 0, 3],
            end_positions_pred[:, 1, 3],
            end_positions_pred[:, 2, 3], 'bo')
    plt.show()
