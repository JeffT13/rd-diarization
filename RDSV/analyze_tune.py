import os, csv, sys, json
import numpy as np
from param import *


with open(embed_path) as je: 
    embed = json.load(je)
  
with open(tune_eval_path) as jt: 
    tune = json.load(jt)
    
for key in embed.keys():
    time = [i[1] for i in embed[key]]
    size = [i[2] for i in embed[key]]
    print('R=', key, '| avg size=', np.mean(size))
    
    run = tune[key]
    for i in run.keys():
        temp = run[i]
        if temp[0][1]<.2:
            print(i,':', temp[0][1], np.sqrt(temp[0][2]), temp[0][0][1])