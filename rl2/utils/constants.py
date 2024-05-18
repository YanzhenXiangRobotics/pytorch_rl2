"""
Global constants for the rl2 python package.
"""
import torch as tc


ROOT_RANK = 0
DEVICE = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
