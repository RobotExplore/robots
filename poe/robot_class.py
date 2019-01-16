# coding: utf-8

import sympy as sp
import numpy as np
import unittest
import timeit
from functools import reduce
from itertools import product
from IPython.core.interactiveshell import InteractiveShell
from IPython import get_ipython

# InteractiveShell.ast_node_interactivity = "all"
# sp.init_printing()


class Robot():

    def __init__(self, axis_params, ini_M):
        self.axis_params = axis_params
        self.ini_M = ini_M

        self.__fk = self.fk_in_space()
        self.fk = lambda joints: self.__fk(
            joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])

        self.__jacobian = self.jacobian_fun()
        self.jacobian = lambda joints: self.__jacobian(
            joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])

    def exp_motion(self, axis_param, angle):
        #     axis_param, angle = params
        # axis_param has 6 elements, first 3 represents the angle velocity, the rest prismatic velocity
        # angle is the rotate angle about revolute joint or displacement along the primatic joint
        w_vector = axis_param[:3, :]
        p_vector = axis_param[3:, :]
        skew_sym_w = sp.Matrix([[0, - w_vector[2], w_vector[1]],
                                [w_vector[2], 0, -w_vector[0]],
                                [-w_vector[1], w_vector[0], 0]])
        R = sp.eye(3) + sp.sin(angle)*skew_sym_w + \
            (1 - sp.cos(angle))*skew_sym_w**2
        G = angle*sp.eye(3) + (1 - sp.cos(angle))*skew_sym_w + \
            (angle - sp.sin(angle))*skew_sym_w**2
        return R.col_insert(3, G*p_vector).row_insert(3, sp.Matrix([[0, 0, 0, 1]]))

    def fk_in_space(self, joints=None, expr_fun=False):
        # in space means the joint axis respect to the fixed frame
        # in body means the joint axis respect to the body frame
        if not joints:
            expr_fun = True  # return the numpy expression function
            w1, w2, w3, w4, w5, w6 = sp.symbols("w1, w2, w3, w4, w5, w6")
            joints = sp.Matrix([w1, w2, w3, w4, w5, w6])
        exps = [self.exp_motion(self.axis_params[i, :].T, joints[i])
                for i in range(6)]
        if expr_fun:
            # this is a function with 6 joint paramters needed
            return sp.lambdify([w1, w2, w3, w4, w5, w6], sp.sympify(sp.prod(exps)*self.ini_M), "numpy")
        else:
            # this is the symbol expression the end position
            return sp.sympify(sp.prod(exps)*self.ini_M)

    def jacobian_fun(self):
        w1, w2, w3, w4, w5, w6 = sp.symbols("w1, w2, w3, w4, w5, w6")
        joints = sp.Matrix([w1, w2, w3, w4, w5, w6])
        jacobian_symb = self.fk_in_space(
            joints)[:3, :].reshape(12, 1).jacobian(joints)
        return sp.lambdify(joints, jacobian_symb, "numpy")

    def ik(self, target_position, current_joints):
        step_size = 1.0
        iter_step = 0
        while True:
            current_position = self.fk(current_joints)
            positon_error = (
                target_position[:3, :] - current_position[:3, :]).reshape(12)
            if np.linalg.norm(positon_error) < 1e-10 or iter_step > 100:
                # print(f"{iter_step}")
                return current_joints
            jacob = self.jacobian(current_joints)
            joints_error = np.linalg.pinv(jacob).dot(positon_error)
            current_joints += joints_error*step_size
            iter_step += 1


class TestRobot(unittest.TestCase):
    def setUp(self):
        # the UR5 robot Product of Exponential parameters
        axis_params = sp.Matrix([[1.1, 0, 1, 0, 0, 0],
                                 [0, 1, 0, -0.089, 0, 0],
                                 [0, 1, 0, -0.089, 0, 0.425],
                                 [0, 1, 0, -0.089, 0, 0.817],
                                 [0, 0, -1, -0.109, 0.817, 0],
                                 [0, 1, 0, 0.006, 0, 0.817]])
        ini_M = sp.Matrix([[-1, 0, 0, 0.817],
                           [0, 0, 1, 0.191],
                           [0, 1, 0, -0.006],
                           [0, 0, 0, 1]])
        self.robot_ur5 = Robot(axis_params, ini_M)

    def test_ik(self):
        # test inverse kinematics
        current_joints = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
        target_joinnts = current_joints + 0.1
        target_position = self.robot_ur5.fk(target_joinnts)

        target_joinnts_ik = self.robot_ur5.ik(target_position, current_joints)
        error = target_joinnts_ik - target_joinnts
        allzero = np.where(error < 1e-8, 0, error).any()
        self.assertFalse(allzero, msg=f"{error}")

    def test_fk(self):
        # test forward kinematics
        # this is more than 10 times faster than the symbol expression
        joints = np.array([0, -np.pi/2, 0, 0, np.pi/2, 2 * np.pi])
        pos1 = self.robot_ur5.fk(joints)
        joints = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
        pos2 = self.robot_ur5.fk(joints)
        error = pos1 - pos2
        error = np.where(error < 1e-8, 0, error).any()
        self.assertFalse(error)

        # # test the symbol expression
        p1, p2, p3, w1, w2, w3, w4, w5, w6 = sp.symbols(
            "p1, p2, p3, w1, w2, w3, w4, w5, w6")
        joints = sp.Matrix([w1, w2, w3, w4, w5, w6])
        pos_symb = self.robot_ur5.fk_in_space(joints)
        self.assertEquals(pos_symb.shape, (4, 4))


if __name__ == "__main__":
    # unittest.main()

    # the UR5 robot Product of Exponential parameters
    axis_params = sp.Matrix([[1.1, 0, 1, 0, 0, 0],
                             [0, 1, 0, -0.089, 0, 0],
                             [0, 1, 0, -0.089, 0, 0.425],
                             [0, 1, 0, -0.089, 0, 0.817],
                             [0, 0, -1, -0.109, 0.817, 0],
                             [0, 1, 0, 0.006, 0, 0.817]])
    ini_M = sp.Matrix([[-1, 0, 0, 0.817],
                       [0, 0, 1, 0.191],
                       [0, 1, 0, -0.006],
                       [0, 0, 0, 1]])
    robot_ur5 = Robot(axis_params, ini_M)
    current_joints = np.array([0, -np.pi/2, 0, 0, np.pi/2, 0])
    target_joinnts = current_joints + 0.1
    target_position = robot_ur5.fk(target_joinnts)

    performance_fk = timeit.timeit(stmt="robot_ur5.fk(current_joints)",
                                   setup="from __main__ import robot_ur5, current_joints", number=100)
    print(f"fk: {performance_fk/100}s")

    performance_ik = timeit.timeit(stmt="robot_ur5.ik(target_position, current_joints)",
                                   setup="from __main__ import robot_ur5, target_position, current_joints", number=100)
    print(f"ik: {performance_ik/100}s")
