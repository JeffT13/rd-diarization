import os, sys, json, random
from math import floor
from param import *
 
cases = os.listdir(audio_path)
r_set = []
d_set = []
t_set = []

for d in train_dockets:
    dock = [a for a in cases if a[:2]==d]
    if seed is not None:
        random.Random(seed).shuffle(dock)
    else:
        random.Random().shuffle(dock)
    r_set.append(dock[:r_count])
    d_set.append(dock[r_count:(r_count+d_count)])
r_set = [item for sublist in r_set for item in sublist]
d_set = [item for sublist in d_set for item in sublist]

split = floor(t_lim/len(test_dockets))
for t in test_dockets:
    dock = [a for a in cases if a[:2]==t]
    if seed is not None:
        random.Random(seed).shuffle(dock)
    else:
        random.Random().shuffle(dock)
    if t_lim is not None and split<len(dock):
        t_set.append(dock[:split])
    else:
        t_set.append(dock)
t_set = [item for sublist in t_set for item in sublist]
    
    
    
set_dict = {'r':r_set, 'd':d_set, 't':t_set}
with open(set_path, 'w') as setfile:  
    json.dump(set_dict, setfile) 
if verbose:
    print('Case Set Dict Saved')


        
        

