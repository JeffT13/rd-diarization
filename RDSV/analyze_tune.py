import os, csv, sys, json
import numpy as np
from scipy import stats
from param import *

avgder_thresh = .25

with open(embed_path) as je: 
    embed = json.load(je)
  
with open(tune_eval_path) as jt: 
    tune = json.load(jt)
    
for key in embed.keys():
    size = [i[2] for i in embed[key]]
    print('R=', key, '| avg size=', np.mean(size))
    
    run = tune[key]
    for r in run.keys():
        ral = run[r][1]
        ev = run[r][0]
        print('RAL settings:', r, '|', ral)
        for set in ev.keys():
            temp = stats.describe(ev[set])
            if temp[2]<avgder_thresh:
                print(i,':', temp[2], np.sqrt(temp[3]), temp[1][1])