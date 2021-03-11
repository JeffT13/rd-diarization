#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Functions to call the oyez API, parse the output and store transcript and s3 links were provided by:
# https://github.com/walkerdb/supreme_court_transcripts
# 
# case_summaries.json was provided by:
# https://github.com/walkerdb/supreme_court_transcripts

import json
from datetime import date
import traceback

import requests

import os
import pandas as pd

# @sleep_and_retry
# @limits(calls=10, period=10)  # no more than 1 call per second
def get_http_json(url):
    print(f"Getting {url}")
    response = requests.get(url)
    parsed = response.json()
    return parsed

def get_case(term, docket):
    """Get the info of the case and fetch all
    transcripts that the info links to"""
    url = f"https://api.oyez.org/cases/{term}/{docket}"
    docket_data = get_http_json(url)

    if not (
        "oral_argument_audio" in docket_data and docket_data["oral_argument_audio"]
    ):
        # no oral arguments for this case yet
        # fail so we will try again later
        print(f"No oral arguments for docket {docket}")
        return (docket_data, [])

    oral_argument_audio = docket_data["oral_argument_audio"]
    transcripts = []
    for link in oral_argument_audio:
        t = get_http_json(link["href"])
        transcripts.append(t)

    return docket_data, transcripts

def getAudio(transcripts):
    num_files = len(transcripts)
    audio_list = []
    for t in transcripts:
        media_dicts = t['media_file']
        if media_dicts[0] is not None: #handle weird cases
            #just incase theres more than one, there shouldnt be but they re in a weird list
            for media_dict in media_dicts:
                audio_list.append(media_dict['href'])
    return [num_files,audio_list]

#gets transcript along with metadata
def getTranscript(transcripts):
    transcript_list = []
    speaker_list = []
    speaker_type_list = []
    time_list = []
    
    #parse through many levels of json file
    for t in transcripts:
        sections = t['transcript']['sections']
        for section in sections:
            turns = section['turns']

            for turn in turns:
                
                #collect speaker
                try:
                    speaker = turn['speaker']['name']
                except:
                    speaker = '<UNK>'
                speaker_list.append(speaker)   
                
                #collect speaker type
                try:
                    roles = turn['speaker']['roles']

                    if isinstance(turn['speaker']['roles'], list):
                        roles = turn['speaker']['roles']
                        multiple_roles = []
                        for role in roles:
                            multiple_roles.append(role['type'])
                        speaker_type_list.append(multiple_roles)

                    else:
                        speaker_type_list.append(['Other']) #Other is most likely Lawyer
                except:
                    speaker_type_list.append(['Other'])
                
                
                #collect text and time
                texts = turn['text_blocks']
                texts_out = []
                times_out = []
                for text in texts:
                    texts_out.append(text['text'])
                    times_out.append((text['start'],text['stop']))
                
                transcript_list.append(texts_out)
                time_list.append(times_out)

    return transcript_list, speaker_list, speaker_type_list, time_list

def main():
    # Get all the terms and dockets from case_summaries.json file
    with open(os.getcwd() + '/case_summaries.json') as f:
        data = json.load(f)

    case_summaries = pd.DataFrame(data)
    case_summaries = case_summaries[['term', 'docket_number']]
    
    # Period of interest
    #test
    #case_summaries_filtered = case_summaries[(case_summaries['term']=='2020')]
    case_summaries_filtered = case_summaries[(case_summaries['term']>='2017') & (case_summaries['term']<'2019')]
    data = {}

    for term, docket_number in case_summaries_filtered.itertuples(index=False):
        docket_data, transcripts = get_case(term, docket_number)
        data[docket_number] = transcripts
        
    # Save cases with only 1 audio file in dictionary 
    audio_data = {}

    for docket, transcript in data.items():
        if bool(data[docket]) and type(data[docket][0]['transcript']) == dict:
            if getAudio(data[docket])[0] == 1 and len(getAudio(data[docket])[1])==1:
                s3_link = getAudio(data[docket])[1][0]
                audio_data[docket] = s3_link

    # Create .shell script for HPC terminal
    file1 = open("mp3_curl_cmds.sh","w") 

    L = ["#!/bin/bash \n",
         "#SBATCH --nodes=1 \n",
         "#SBATCH --ntasks-per-node=1 \n",
         "#SBATCH --cpus-per-task=1 \n",
         "#SBATCH --time=5:00:00 \n",
         "#SBATCH --mem=2GB \n",
         "#SBATCH --job-name=get_oyez_mp3s \n",
         "\n"]  
    file1.writelines(L) 

    for docket, s3_link in audio_data.items():
        file1.write(f'curl -L {s3_link} -o {docket}.mp3 \n')

    file1.close() 
    
    print("mp3_curl_cmds.sh created.")

    mp3_meta_data = {}

    # Using transcript_data_clean.keys() to get the list of dockets from 2011 - 2020 that:
    #   1. All have transcripts 
    #   2. All have just 1 mp3 file 
    for docket in audio_data.keys():
        print(docket)
        transcript_list, speaker_list, speaker_type_list, time_list = getTranscript(data[docket])
        mp3_meta_data[docket] = transcript_list, speaker_list, speaker_type_list, time_list

    with open('oyez_metadata.json', 'w+') as f:
        # this would place the entire output on one line
        # use json.dump(lista_items, f, indent=4) to "pretty-print" with four spaces per indent
        json.dump(mp3_meta_data, f)

    print("oyez_metadata.json file created.")
    
    return print("Done!")

main()
    
