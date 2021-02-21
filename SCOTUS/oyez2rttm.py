import os, sys, json
sys.path.append('/home/jt2565/diar/rd-diarization')
from RDSV.param import *


label = 20 # unknown speaker label counter (leave room for 20 judges)
process_sd=False

if os.path.exists(sd_path):
  with open(sd_path) as json_file: 
    spkr_dict = json.load(json_file)
    cnt = max([v for v in spkr_dict.values() if v<label])
    label = max(spkr_dict.values())
  print('Loaded spkr dictionary')
else:
  print('build new spkr dictionary')
  spkr_dict = dict()
  cnt = 0
  process_sd=True

for filename in os.listdir(trans_path):
    case_name = filename.split('.')[0]
    #check if it has been ran successfully
    if os.path.exists(rttm_path+case_name+'.rttm'):
      print('Case', case_name, 'already processed')
      continue
    else:
      print('Processing Case', case_name)
      f = open(trans_path+filename,'r')
      k = f.readlines()
      f.close()

      timelst = []
      for u in k:
        t0, t1, spkr = u.split(' ')[0:3]
        #build speaker dictionary
        if spkr[-14:]=='scotus_justice':
            if spkr not in spkr_dict:
              spkr_dict[spkr] = cnt
              cnt+=1
              if cnt>=20:
                print("ERROR: NEED MORE JUDGE ROOM")
        else:
            if spkr not in spkr_dict:
              spkr_dict[spkr] = label
              label+=1       
        timelst.append((float(t0),float(t1),spkr))
        
      torttm = []
      for i, event in enumerate(timelst):
        torttm.append(' '.join(['SPEAKER '+case_name+' 1', str(event[0]), str(round(event[1]-event[0], 2)), '<NA> <NA>', event[2],'<NA> <NA>']))
        
      with open(rttm_path+case_name+'.rttm', 'w') as filehandle:
          for listitem in torttm:
              filehandle.write('%s\n' % listitem)
if process_sd:
  print('New Spkr Dict Saved')
  with open(sd_path, 'w') as outfile:  
    json.dump(spkr_dict, outfile) 



#Generate random case sets
if os.path.exists(audio_path):
    cases = os.listdir(audio_path)
    if seed is not None:
        random.Random(seed).shuffle(cases)
    else:
        random.Random().shuffle(cases)
        
    c_set = cases[:c_set_count]
    d_set = cases[c_set_count:(d_set_count+c_set_count)]
    if t_lim is not None:
        t_set = cases[d_set_count:t_lim]
    else:
        t_set = cases[d_set_count:]
    
    set_dict = {'c':c_set, 'd':d_set, 't':t_set}
    with open(fp+'set_file.json', 'w') as setfile:  
        json.dump(set_dict, setfile) 
    print('Case Set Dict Saved')

else:
    print('File paths do not exist')
