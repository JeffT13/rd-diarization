import os, csv, sys, json
import numpy as np
from scipy import stats
from param import *

avgder_thresh = .20
hold = 100
with open(embed_path) as je: 
    embed = json.load(je)
  
with open(tune_eval_path) as jt: 
    tune = json.load(jt)
    
for key in embed.keys():
    size = [i[2] for i in embed[key]]
    print('\n\n\n\n\n')
    print('R=', key, '| avg size=', np.mean(size))
    
    run = tune[key]
    for r in run[0].keys():
        ral = run[1][r]
        ev = run[0][r]
        print('RAL settings:', r, '|', ral)
        print()
        for set in ev.keys():
            temp = stats.describe(ev[set])
            if temp[2]<avgder_thresh:
                #Mean, SD, Max
                print(r,'-',set,':', temp[2], np.sqrt(temp[3]), temp[1][1])
                if temp[2]<hold:
                    hold=temp[2]
                    id = key+'|'+r+'|'+set
                    perm = temp
                    
print('\n\n\n\n', 'Best Param:', id, ' === ', perm)