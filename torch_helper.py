"""
This file contains basics helper function for pyTorch models used by colab NB (e.g. conversion of numpy arrays to tensors, building out training and testing loops, etc.)
"""
import torch


def to_tensor(arr):
    """
    Convert a numpy array to a pyTorch tensor
    """
    return torch.tensor(arr, dtype=torch.float32)
