
# coding: utf-8

# In[1]:


import sympy as sp
import numpy as np
from functools import reduce
from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"
# sp.init_printing()


# In[12]:


p1, p2, p3, w1, w2, w3, w4, w5, w6 = sp.symbols(
    "p1, p2, p3, w1, w2, w3, w4, w5, w6")

# w = sp.Matrix([w1, w2, w3])
# w.norm(2)

# m = sp.Matrix([w.T, w.T, w.T])
# sp.matrix2numpy(w.T)

# # def skew_symmetrical(w_vector):
# #     """
# #     param: w_vector has 3 elements
# #     """
# #     return sp.Matrix([[0, - w_vector[2], w_vector[1]],
# #                     [w_vector[2], 0, -w_vector[0]],
# #                     [-w_vector[1], w_vector[0], 0]])


# In[17]:


def exp_motion(axis_param, angle):
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


def fk_in_space(joints=None, expr=False):
    # in space means the joint axis respect to the fixed frame
    # in body means the joint axis respect to the body frame
    if not joints:
        expr = True  # return the numpy expression function
        w1, w2, w3, w4, w5, w6 = sp.symbols("w1, w2, w3, w4, w5, w6")
        joints = sp.Matrix([w1, w2, w3, w4, w5, w6])
    axis_params = sp.Matrix([[0, 0, 1, 0, 0, 0],
                             [0, 1, 0, -0.089, 0, 0],
                             [0, 1, 0, -0.089, 0, 0.425],
                             [0, 1, 0, -0.089, 0, 0.817],
                             [0, 0, -1, -0.109, 0.817, 0],
                             [0, 1, 0, 0.006, 0, 0.817]])
    ini_M = sp.Matrix([[-1, 0, 0, 0.817],
                       [0, 0, 1, 0.191],
                       [0, 1, 0, -0.006],
                       [0, 0, 0, 1]])
    exps = [exp_motion(axis_params[i, :].T, joints[i]) for i in range(6)]
    if expr:
        # this is a function with 6 joint paramters needed
        return sp.lambdify([w1, w2, w3, w4, w5, w6], sp.sympify(sp.prod(exps)*ini_M), "numpy")
    else:
        # this is the symbol expression the end position
        return sp.sympify(sp.prod(exps)*ini_M)


# In[18]:


# the UR5 robot
# axis_params = sp.Matrix([[1.1, 0, 1, 0, 0, 0],
#                          [0, 1, 0, -0.089, 0, 0],
#                          [0, 1, 0, -0.089, 0, 0.425],
#                          [0, 1, 0, -0.089, 0, 0.817],
#                          [0, 0, -1, -0.109, 0.817, 0],
#                          [0, 1, 0, 0.006, 0, 0.817]])
# ini_M = sp.Matrix([[-1, 0, 0, 0.817],
#                   [0, 0, 1, 0.191],
#                   [0, 1, 0, -0.006],
#                   [0, 0, 0, 1]])
joints = sp.Matrix([w1, w2, w3, w4, w5, w6])
# joints = sp.Matrix([0]*6)
# joints[1] = - sp.pi / 2
# joints[4] = sp.pi / 2


# In[19]:


fk = fk_in_space()


# In[20]:


get_ipython().run_line_magic('timeit', 'fk(0, -np.pi/2, 0, 0, np.pi/2, 0)')


# In[21]:


get_ipython().run_line_magic(
    'timeit', 'fk_in_space(sp.Matrix([0, -sp.pi/2, 0, 0, sp.pi/2, 0]))')
