# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:30:06 2022

@author: LarsNelson
"""

import matplotlib.transforms as tx
import numpy as np


trans = tx.blended_transform_factory(ax.transData, ax.transAxes)
xy_pairs = np.column_stack([
    np.repeat(1, 2), np.tile([0, 1], np.arange(1,30, 0.1))
])