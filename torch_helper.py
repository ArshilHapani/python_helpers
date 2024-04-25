"""
This file contains basics helper function for pyTorch models (e.g. conversion of numpy arrays to tensors, building out training and testing loops, etc.)
"""


def to_tensor(arr):
    """
    Convert a numpy array to a pyTorch tensor
    """
    return torch.tensor(arr, dtype=torch.float32)