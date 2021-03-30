# SCOTUS Speaker Verification

## SCOTUS Preprocessing Details

Prepping SCOTUS data for modeling. 

Credit: 
- https://www.oyez.org/ for their API archive of Supreme Court multimedia
- https://github.com/walkerdb/supreme_court_transcripts for oyez parsing functions in mp3-to-gcp/oyez_parser.ipynb

### Prerequisites 

You will need the case_summaries.json from https://github.com/walkerdb/supreme_court_transcripts/tree/master/oyez.

Most of our computing is done on an HCP that does not support the requests package and therefore some of the scripts have been split to run locally and transfer of necessary files are done manually. 

All the modules in HPC needed for this process are:

module purge 

module load ffmpeg/4.2.4 

module load python/intel/3.8.6

module load rclone/1.53.3 

### Data Pulling Instructions

1. Extracting mp3 files and metadata from oyez API with **mp3_curl_commands.py**
- NOTE: Our HPC cluster does not support the requests package, so this step happens locally. 
- NOTE: This script extracts cases for which there is only *one* mp3 file for the corresponding case. 
- Have case_summaries from walkerdb's github in the same directory as the mp3_curl_commands.py script
- Specify date range for cases you want in mp3_curl_commands.py (line 116)
- Run with `python mp3_curl_commands.py`
- Output is: shell script mp3_curl_cmds.sh that you can run in HPC to get case mp3 from oyez API && oyez_metadatajson which we need in step 5. 

2. Run **mp3_curl_cmds.sh** 
- NOTE: This was performed in HPC 
- Run with `sbatch mp3_curl_cmds.sh`

3. Convert mp3s to wav files with **mp3_to_wav_batch.sh**
- NOTE: This was performed in HPC 
- NOTE: You might want to create a wavs folder to run the script in first. You can `mv audio_split.py wavs`, `mv mp3_to_wav_batch.sh wavs`, `mv make_transcripts.py wavs` and `mv oyez_metadata.json wavs`.
- Install current versions of ffmpeg with `module load ffmpeg/intel/3.2.2`
- Run with `sbatch mp3_to_wav_batch.sh /path/to/files /path/to/dest` EXAMPLE: `sbatch mp3_to_wav_batch.sh /scratch/smt570/test/mp3s /scratch/smt570/test/wavs`

4. Pull Transcriptions with **make_transcripts.py**
- NOTE: This was performed in HPC 
Run `python make_transcripts.py`. Make sure this file is sitting in the same folder with all the wav files (`mv make_transcripts.py wavs` if you haven't already.) 


### Transcription to RTTM

Simply run the `oyez2rttm.py` script to write the transcriptions to the `rttm` folder in RTTM format. This also generates and saves a dictionary of present speakers and identification numbers so that each speaker is labelled consistently throughout the experiments. 