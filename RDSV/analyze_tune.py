import os, csv, sys, json
import numpy as np
from scipy import stats
from param import *


avgder_thresh = .3
hold = 100

with open(tune_eval_path) as jt: 
    tune = json.load(jt)


for key in tune:
    print('RAL settings:', key)
    for k in tune[key]:
        print('diar settings:', k)
        temp = stats.describe(tune[key][k])
        if temp[2]<avgder_thresh:
            #Mean, SD, Max
            print(round(temp[2],3), round(np.sqrt(temp[3]),3), round(temp[1][1], 3))
            if temp[2]<hold:
                hold=temp[2]
                id = key+'|'+k
                perm = temp
    print('\n\n')
    
print('\n\n\n', 'Best Param:', id, ' === ', perm)

    

    
