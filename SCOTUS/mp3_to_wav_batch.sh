#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=mp3_to_wav_batch

# Credit: https://linuxconfig.org/how-to-use-ffmpeg-to-convert-multiple-media-files-at-once-on-linux
  
module purge
module load ffmpeg/4.2.4

srcExt=mp3
destExt=wav

srcDir=$1
destDir=$2

opts='-acodec pcm_s16le -ac 1 -ar 16000'

for filename in "$srcDir"/*.$srcExt; do

        basePath=${filename%.*}
        baseName=${basePath##*/}

        ffmpeg -i "$filename" $opts "$destDir"/"$baseName"."$destExt"

done

echo "Conversion from ${srcExt} to ${destExt} complete!"
