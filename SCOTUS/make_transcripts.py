#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import os

import pandas as pd

def getMeta(docket, data):
    
    #get meta data as well as rearrange to desirable formal
    transcript, speakers, speaker_roles, times = data[docket]
    
    # Flatten times list
    times_new = []
    for t in times:
        flatten = [item for sublist in t for item in sublist]
        times_new.append(flatten)  
    # Last element of list is a 0 - cleanup    
    del times_new[-1][-1]
    
    # Flatten speaker_roles list and replace nulls with "Other"
    speaker_roles_clean = []
    for i in speaker_roles:
        if not i:
            speaker_roles_clean.append('Other')
        else:
            speaker_roles_clean.append(i[0])
            
    # Remove all non-word characters in speakers' names
    speakers =[re.sub(r"[^\w\s]", '', s) for s in speakers]
    # Replace all runs of whitespace with underscorei in speakers' names
    speakers =[re.sub(r"\s+", '_', s) for s in speakers]
    
    return transcript, speakers, speaker_roles_clean, times_new

def main_script():
    
    # Get oyez meta data for all dockets out there
    with open(os.getcwd() + '/oyez_metadata.json') as f:
        data = json.load(f)
    
    # Get names of dockets for which we have wav files for in cwd
    saved_dockets = []
    for file in os.listdir(os.getcwd()):
        if file.endswith(".wav"):
            saved_dockets.append(file.split('.')[0])
    
    # Create transcript for wav files saved if certain criteria check out 
    for docket in saved_dockets:
        transcript, speakers, speaker_roles, times_new = getMeta(docket,data)
        
        if len(transcript) == len(speakers) == len(speaker_roles) == len(times_new):
            with open(f"{os.getcwd()}/{docket}.txt", "w") as outfile:
                case_info = pd.DataFrame([times_new,speakers,speaker_roles,transcript]).T

                for i in range(len(case_info)):
                    start = case_info.iloc[i][0][0]
                    stop = case_info.iloc[i][0][-1]
                    speaker = case_info.iloc[i][1] + '_' + case_info.iloc[i][2]
                    speaker_text = " ".join(case_info.iloc[i][3])

                    # Write .txt
                    outfile.write(str(start) + ' ' + str(stop) + ' ' + speaker + ' ' + speaker_text + '\n')
    

main_script()
