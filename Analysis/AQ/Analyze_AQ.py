#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:35:32 2021

@author: ravinderjit
"""
import pandas as pd
import os

data_loc = '/media/ravinderjit/Data_Drive/Data/AQ/'
data_file = 'AQ_October 19, 2021_15.25.csv'

data = pd.read_csv(os.path.join(data_loc,data_file))

answers = ['Definitely Agree', 'Slightly Agree', 'Slightly Disagree', 'Definitely Disagree']


Subjects = data['1a'][2:].to_numpy()



