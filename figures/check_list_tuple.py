#!/usr/bin/env python3
import numpy as np

# Checks if something is a list or a tuple

def check_list_tuple(v):
    """
    Checks to see if something is a list or tuple
    """
    if type(v) is list:
        return True
    if type(v) is tuple:
        return True
    # implicit else
    return False


def check_numpy_type(v):
    """
    Returns True if module is a numpy type
    Returns False if module is not numpy type
    """
    if type(v).__module__ == np.__name__:
        return True
    else:
        return False
