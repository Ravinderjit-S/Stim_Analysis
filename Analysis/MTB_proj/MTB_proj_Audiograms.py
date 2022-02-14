#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 17:11:12 2022

@author: ravinderjit
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data_loc_Aud = '/media/ravinderjit/Data_Drive/Data/MTB_Behavior/Audiograms/BekAud.mat'

aud = sio.loadmat(data_loc_Aud,squeeze_me=True)

plt.figure()
plt.plot(aud['f_fit'], aud['audiogram_R'])

