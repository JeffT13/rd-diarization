# rd-diarization

Code repository for paper [**Diarization of Legal Proceedings. Identifying and Transcribing Judicial Speech from Recorded Court Audio**](https://arxiv.org/abs/2104.01304)

We focus on the task of audio diarization in the legal domain, specifically on the Supreme Court of the United States Oral Arguments, accessed through the [Oyez Project](https://www.oyez.org/). In this work utilize a [speech embedding network](https://github.com/resemble-ai/Resemblyzer) (referred to as *Voice Encoder*) pre-trained by Resemble.AI with the [Generalized End-to-End Loss](https://arxiv.org/abs/1710.10467) to encode speech into d-vectors and a pre-defined reference audio library based on annotated data. We find that by encoding reference audio for speakers and full proceedings and computing similarity scores we achieve a 13.8% Diarization Error Rate for speakers covered by the reference audio library on a held-out test set.

## Repository Contents

The documentation outlines reproducing our experiments on the SCOTUS oral arguments. Most computation was done through SLURM jobs on a cluster but some stages are done locally, and all can be if you have a GPU or perform the audio embedding on cpu, which is fine for short audio. The process involves data preperation, detailed in the `SCOTUS` folder, and embedding and diarization, outlined in the `RDSV` folder. There is the `script` folder which holds the script calls as SLURM jobs, and in our `plots` folder there is a notebook analyzing the final results and generating plots, which was ran locally.

The `SCOTUS` folder contains the code to pull case audio and transcriptions from the [Oyez API](https://github.com/walkerdb/supreme_court_transcripts),convert the mp3 to wav files (uses `ffmpeg`), as well as convert the transcriptions to RTTM format. The `RDSV` folder houses the case embedding and diarization code, as well as [our fork of the Voice Encoder](https://github.com/JeffT13/VoiceEncoder) as a submodule. Each of these folders have their own README detailing their content. Finally, there is the `script` folder which holds the python script calls as SLURM jobs, and in our `plots` folder there is a notebook analyzing the final results and generating plots, which was ran locally. 


## Set-Up

After cloning this repository, make sure to fetch the files for the Voice Encoder submodule (`git submodule update --init`). Our implementation assumes you will be running your jobs from directly inside the git repo and have a `data` directory outside of this repo which contains an `audio`,`transcript` and `rttm` folder. This can be changed in the `RDSV/param.py` file. The Voice Encoder also has it's own parameter file, which we do not change from the original, but can be altered. Once this is complete and enviornment has been created, you are ready to reproduce our experiments. Start by following the documentation in the `SCOTUS` folder.

    
### Requirements

We used python version `3.7.6`. You can build your env with the `requirements.txt` file. The only packages that are outside of typical data science packages are:

    - PyTorch
    - librosa
    - webrtcvad
    - pyannote.audio
  
We You can run the `RDSV/env_check.py` script to ensure the enviorment has been set up correctly.


# Experiment Results

## Voice Encoder Analysis

We embedded a set of cases for a range of rates and performed multiclass one-vs-rest Logistic Regression to infer d-vector speaker labels. We train the classifier on only the d-vectors from the judicial speech found in the R set audio. We then evaluate the ensemble classifier's performance on the d-vectors from the full proceedings in our D set. We do this by thresholding the predicted probability from the classifier to predict a non-judge speaker for a range of values L=[.85, .9, .95]. We found that a rate of $5$ had the most stable accuracy as the threshold increased and the highest overall accuracy for all thresholds so we set our \code{rate}$=5$ for our diarization process.

#### Logistic Regression Performance

| rate | L=.85 | L=.9 | L=.95 |
|------|-------|------|-------|
| 1    | .84   | .79  | .65   |
| 3    | .82   | .83  | .79   |
| 5    | .80   | .83  | .83   |
| 7    | .78   | .82  | .83   |

#### d-vector tSNE Visualization

We also visualize the d-vectors representing judicial speech from the cases in our R set for our lowest and optimal rate. We reduce our d-vectors to two dimensions using sklearn's t-distributed Stochastic Neighbor Embedding (t-SNE) with a perplexity of 50 (default parameters otherwise). We then plot the shrunken d-vectors with their associated speaker labels so we can see how well our embeddings tend to cluster and if our d-vector labels are accurate. We can see that even our lowest rate performs reasonably, but there is significantly higher ratio of dispersed d-vectors in comparison to the higher rate embeddings.

![](./plots/dvector_tsne.png)

## RDSV Performance

The table shows the diarization performance of our tuned pipeline on the 25 cases in the test set. We also show a 10 minute snippet of a groundtruth case diarization (top) and our inferred diarization (bottom). This is only considered the diarization of speakers that are part of the reference set.

| Metric              | Value    |
|---------------------|----------|
| Average DER         | 13.8%    |
| DER Std Dev         | 3.5%     |
| Max DER             | 19.8%    |
| Audio/Case          | 58.9 min |
| Audio/Case post-VAD | 51.4 min |

![](./plots/segplot.png)

## Acknowledgements

Tremendous thanks to Professor Aaron Kaufman, Professor Michael Picheny and Professor Brian McFee for their contributions to the paper as well as their guidance and feedback on the implementation. We also thank New York University's High-Performance Computing Clusters for the computational resources and Dr. Sergey Samsonau for his help in utilizing these resources. Finally, we thank Resemble.AI for their open-source contributions to speech research and the Oyez Project for the service they provide in making Supreme Court data available. 
