import os, csv
from VoiceEncoder import audio
from VoiceEncoder.voice_encoder import VoiceEncoder #sorry for this

def getDiary(file_path):
  with open(file_path, newline='\n') as f:
    reader = csv.reader(f)
    case_diary = list(reader)
  return case_diary 

def casewrttm_to_dvec(audio_path, rttm_path, sd_path, device, sr=16000, verbose=True):
  #preprocess wav file
  wav, labels, mask = audio.preprocess_wav(audio_path, rttm_path, sd_path, source_sr=sr) #labels are case preset currently
  #call model
  encoder = VoiceEncoder(device)
  if verbose:
    print("Running the continuous embedding for "+str(audio_path).split('/')[-1]+"...")
  #create dvectors
  embed, splits = encoder.embed_utterance(wav, mask[-1], wav_labels=labels, sd_path=sd_path, verbose=verbose)
  if verbose:
    print(np.shape(embed[0]), np.shape(embed[1]), np.shape(embed[2]))
  return embed, splits, (wav, labels), mask
  
def case_to_dvec(audio_path, device, sr=16000, verbose=True):
  #preprocess wav file
  wav, mask = audio.preprocess_wav(audio_path, source_sr=sr) #labels are case preset currently
  #call model
  encoder = VoiceEncoder(device)
  if verbose:
    print("Running the continuous embedding for "+str(audio_path).split('/')[-1]+"...")
  #create dvectors
  embed, info = encoder.embed_utterance(wav, mask, verbose=verbose)
  if verbose:
    print(np.shape(embed))
  return embed, info

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
