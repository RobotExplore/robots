import numpy as np
from functools import reduce

AXIS = ('X', 'Y', 'Z')


def rotate(axis, deg):
    axis = str(axis).upper()
    if axis not in AXIS:
        print(f"{axis} is unknown axis, should be one of {AXIS}")
        return
    rot_x = axis == 'X'
    rot_y = axis == 'Y'
    rot_z = axis == 'Z'
    rot_mat = np.array([[(np.cos(deg), 1)[rot_x], (0, -np.sin(deg))[rot_z], (0, np.sin(deg))[rot_y], 0],
                        [(0, np.sin(deg))[rot_z], (np.cos(deg), 1)[rot_y], (0, -np.sin(deg))[rot_x], 0],
                        [(0, -np.sin(deg))[rot_y], (0, np.sin(deg))[rot_x], (np.cos(deg), 1)[rot_z], 0],
                        [0, 0, 0, 1]], dtype=np.float32)
    rot_mat = np.where(np.abs(rot_mat) < 1e-10, 0, rot_mat)  # get a small value when np.cos(np.pi/2)
    return rot_mat


def trans(axis, dis):
    axis = str(axis).upper()
    if axis not in AXIS:
        print(f"{axis} is unknown axis, should be one of {AXIS}")
        return
    trans_mat = np.eye(4)
    trans_mat[AXIS.index(axis), 3] = dis
    return trans_mat


def fk(joints):
    """
    :param DH_pramater: [link, a, d, thea]
    :return:
    """
    thea_1, thea_2, thea_3, thea_4, thea_5, thea_6 = joints
    DH = [[thea_1, 399.1, 0, -np.pi/2],
          [thea_2, 0, 448, 0],
          [thea_3, 0, 42, -np.pi/2],
          [thea_4, 451, 0, np.pi/2],
          [thea_5, 0, 0, -np.pi/2],
          [thea_6, 82, 0, 0]]
    T = [rotate('z', thea_i).dot(trans('z', d_i)).dot(trans('x', l_i)).dot(rotate('x', a_i))
         for thea_i, d_i, l_i, a_i in DH]
    T60 = reduce(np.dot, T)
    return  T60


if __name__ == '__main__':
    # assert (rotate('x', np.pi / 2) == np.array(
    #     [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])).all(), 'rotate wrong, x'
    # assert (rotate('Y', np.pi / 2) == np.array(
    #     [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])).all(), 'rotate wrong, y'
    # assert (rotate('z', np.pi / 2) == np.array(
    #     [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])).all(), 'rotate wrong, z'
    # assert (trans('x', 1) == np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])).all(), 'trans wrong'

    """
    # 图中末端的位姿矩阵为：
    [[0, 0, 1, 533], 
    [0, 1, 0, 0], 
    [-1, 0, 0, 889.1], 
    [0, 0, 0, 1]]
    各个关节的取值为：[0, -np.pi/2, 0, 0, 0, np.pi]
    """
    joints = [0, -np.pi/2, 0, 0, 0, np.pi]
    # 打印运动学计算出来的坐标和根据图中几何计算出来的坐标是否相同包
    print(fk(joints))

