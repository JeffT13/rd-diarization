import json
import numpy as np
from VoiceEncoder.util import getDiary

class RefAudioLibrary:
  def __init__(self, c, embed_path, rttm_path, spkrdict_path, judge_only=True, min_audio_len = 4, min_ref_thresh=8, save=False, unref_constant = 'UnRefSpkr'):
  
    self.case_set = c
    self.embed = embed_path
    self.rttm = rttm_path
    self.judge_only = judge_only
    self.uid = unref_constant
    with open(spkrdict_path) as json_file: 
      self.sd = json.load(json_file)

    load_embed = {}
    spkr_ints = {}
    RAL = {}

    #process RTTM & Embeddings (preprocessed)
    for case in self.case_set:
      temp_diary = getDiary(self.rttm+case+'.rttm')
      load_embed[case] = [np.load(embed_path+case+'_embeds.npy'), np.load(embed_path+case+'_embeds_times.npy')]
      for i in range(len(temp_diary)):
        speaker = temp_diary[i][0].split(' ')[7]
        if speaker not in spkr_ints:
          spkr_ints[speaker] = [temp_diary[i]]
        else:
          spkr_ints[speaker].append(temp_diary[i])
    self.spkr_ints = spkr_ints
    self.load_embed = load_embed

    #Build RAL
    for spkr in spkr_ints.keys():
      count = 0
      if judge_only:
        if self.sd[spkr]<20: 
          while count < len(spkr_ints[spkr]):
            entry = spkr_ints[spkr][count][0].split(' ')
            count+=1
            if float(entry[4])>min_audio_len:
              start = float(entry[3])
              end = start + float(entry[4])
              case = entry[1]
              if spkr in RAL:
                RAL[spkr].append(self.get_interval_embed(start, end, load_embed[case][0], load_embed[case][1]))
              else:
                RAL[spkr] = [self.get_interval_embed(start, end, load_embed[case][0], load_embed[case][1])]
      else:
        while count < len(spkr_ints[spkr]):
          entry = spkr_ints[spkr][count][0].split(' ')
          count+=1
          if float(entry[4])>min_audio_len:
            start = float(entry[3])
            end = start + float(entry[4])
            case = entry[1]
            if spkr in RAL:
              RAL[spkr].append(self.get_interval_embed(start, end, load_embed[case][0], load_embed[case][1]))
            else:
              RAL[spkr] = [self.get_interval_embed(start, end, load_embed[case][0], load_embed[case][1])]

    
    if min_ref_thresh is not None:
        sl = []
        for spkr in RAL.keys():
            if len(RAL[spkr])<min_ref_thresh:
                sl.append(spkr)
        for s in sl:
          RAL.pop(s)
                
    self.RAL = RAL
    if save is not False:
      with open(save, 'w') as outfile:  
        json.dump(RAL, outfile) 


  def get_interval_embed(self, start, end, cont_embeds, embed_times):
    '''Takes as input start and end time of speaking interval (in s) as well as embeddings for the full case and corresponding 
    times for each embedding (in ms)
    Returns: single reference audio embedding by performing L2 normalization on average of embeddings across embedding interval'''
    #get starting embed by choosing closest to given start time
    diff = 10
    for i in range(len(embed_times)):
      check = abs(start-embed_times[i][0]/1000)
      if check < diff:
        diff = check
        start_emb = i
    
    #get ending embed by choosing closest to given end time
    diff = 10
    for j in range(len(embed_times)):
      check = abs(end-embed_times[j][1]/1000)
      if check < diff:
        diff = check
        end_emb = j+1
    #take average embedding across embedding interval
    raw_embed = np.mean(cont_embeds[start_emb:end_emb], axis=0)
    #L2 normalize
    embed = raw_embed/np.linalg.norm(raw_embed,2)
    return embed

def Diarize(ral, cont_embeds, embed_times, thresh = 0.1, min_seg = 2):

  '''Takes as input a dictionary of multiple reference embeddings for each speaker, as well as a list of their names. 
  Additionally takes in embeddings of the full court hearing and time intervals associated with each embedding.

  thresh = 0.15 as default - > indicates for each embedding time step that the difference between the highest sim score
  and next highest sim score cannot be less than 0.15, otherwise label as non-judge speech

  small_seg = 1 second as default - > indicates that the shortest speaking times cannot be less than this duration

  Returns: diarization with speaker name and times'''

  #creates similarity score for each reference audio with each case embedding at every timestep
  similarity ={}
  for speaker in ral.RAL.keys():
    for i in range(len(ral.RAL[speaker])):
      if speaker in similarity:
          similarity[speaker].append(cont_embeds @ ral.RAL[speaker][i])
      else:
        similarity[speaker] = [cont_embeds @ ral.RAL[speaker][i]]

  #create similarity dict that stores max val for each speaker at each timestep
  similarity_max = {}
  for speaker in ral.RAL.keys():
    similarity_max[speaker]=np.max(similarity[speaker],axis=0)

  #compute speaker with highest similarity score at each interval
  #if diff between highest sim score and next highest < thresh, label as "Non-Judge"
  speak = []
  sim = []
  next_sim = []
  for i in range(len(cont_embeds)):
    max_sim=0
    next_max_sim = 0
    for name in ral.RAL.keys():
      similarity_score = similarity_max[name][i]
      if similarity_score > max_sim:
        max_sim = similarity_score
        max_name = name
      elif similarity_score > next_max_sim:
        next_max_sim = similarity_score
    
    if (max_sim-next_max_sim) < thresh:
      speak.append(ral.uid)
    else:
      speak.append(max_name)

  #####
  #get diarized list of speaker with actual speaking times and merge if next speaker is curr speaker
  diarized = []
  curr = speak[0]
  start = round(embed_times[0][0]/(1000),2)
  for i in range(len(speak)-1):
    if curr == speak[i+1]:
      continue
    else:
      diarized.append([curr,start,round(embed_times[i][1]/(1000),2)])
      curr = speak[i+1]
      start = round(embed_times[i][1]/(1000),2)

  #for last iteration 

  #if last iteration speaker is same as very last speaker just append
  if curr == speak[i+1]:
    diarized.append([curr,start,round(embed_times[i][1]/(1000),2)])

  #otherwise split into last iteration speaker and very last speaker
  else:
    diarized.append([curr,start,round(embed_times[i][1]/(1000),2)])
    curr = speak[i+1]
    start = round(embed_times[i][1]/(1000),2)
    diarized.append([curr,start,round(embed_times[i+1][1]/(1000),2)])

  #####

  #####
  #remove all small intervals less than min_seg indicated, and merge speaking durations
  diarized_merged = []
  curr = diarized[0][0]
  start = diarized[0][1]
  end = diarized[0][2]

  for i in range(len(diarized)-1):
    if (diarized[i+1][2]-diarized[i+1][1])<min_seg:
      end = diarized[i+1][2]
    else:
      if curr == diarized[i+1][0]:
        end = diarized[i+1][2]
      else:
        diarized_merged.append([curr,start,end])
        start = end
        end = diarized[i+1][2]
        curr = diarized[i+1][0]

  #for last iteration
  diarized_merged.append([curr,start,end])

  #####

  return diarized_merged 



def diar_to_rttm(diar, case, out_path, verbose=True):
  torttm = []
  for i, event in enumerate(diar):
    torttm.append(' '.join(['SPEAKER '+case+' 1', str(event[1]), str(round(event[2]-event[1], 2)), '<NA> <NA>', event[0],'<NA> <NA>']))
  with open(out_path+case+'_rdsv.rttm', 'w') as filehandle:
      for listitem in torttm:
          filehandle.write('%s\n' % listitem)
  if verbose:
    print('Case', case, 'diarization saved')


def rttmto_RALrttm(case, ral, in_path, out_path, verbose=True):
    out_diary = []
    spkr_tracker = []
    diary = getDiary(in_path+case+'.rttm')
    for entry in diary:
      counter = 0
      temp = entry[0].split(' ')
      if temp[7] not in spkr_tracker:
        spkr_tracker.append(temp[7])
      if temp[7] not in ral.RAL.keys():
        temp[7] = ral.uid
        counter+=1
      out_diary.append(' '.join(temp))
    with open(out_path+case+'_ral.rttm', 'w') as filehandle:
        for listitem in out_diary:
            filehandle.write('%s\n' % listitem)
        if verbose:
          print(case, 'rttm has been RAL converted')
          print(len(spkr_tracker), 'total speakers')
          print([s for s in spkr_tracker if s not in ral.RAL.keys()], 'were unreffed')          
