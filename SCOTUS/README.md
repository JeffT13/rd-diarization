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

### Instructions
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

4. Audio Splitting with **audio_split.py**
- NOTE: This is run to convert the wav files into consumable format for our model
- Load the right modules in HPC with: 

`module purge `

`module load ffmpeg/intel/3.2.2 `

`module load python3/intel/3.7.3 `

- NOTE: Make sure you have in the directory with the script (1) the wav files you want to split (2) oyez_metadata.json file from step 1 (3) an empty SCOTUS file and 
- Run with `python audio_split.py`

5. Transcribing WITHOUT splitting with **make_transcripts.py**
- NOTE: This was performed in HPC 
Alternatively, you can create transcripts for your wav files without splitting them by speaker by running `python make_transcripts.py`. Make sure this file is sitting in the same folder with all the wav files (`mv make_transcripts.py wavs` if you haven't already.) 


## d-vector embedding

This procedure outputs a folder for each case processed, each containing a numpy array of dvector embeddings (`case_sequence.npy`) and a numpy array of labels (`case_cluster_id.npy`) which are both the same length, as well as a csv of a list of files which were not embedded (usually do to being too short). These are formatted for [our fork of the uisrnn](https://github.com/JeffT13/uis-rnn) 
