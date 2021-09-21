import numpy as np


def normalized_ratio(a, b):
    return (a - b) / (a + b)


def rescale_s2(arr):
    if arr.dtype == np.uint16 or np.nanmax(arr) > 200 and arr.dtype != np.uint8 and np.nanmax(arr) != 255:
        return arr / 10000  # for Sentinel-2 data directly from ESA
    # not yet rescaled (5 is arbitrary, shouldn't be > 1)
    elif arr.dtype == np.uint8 and np.nanmax(arr) > 5 or np.nanmax(arr) == 255:
        return arr / np.iinfo(np.uint8).max  # for Sentinel-2 from Sentinel Hub (0-255)
    else:
        return arr
