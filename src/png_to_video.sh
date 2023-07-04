#!/bin/bash

input_folder="$1"
output_file="$2"

# Check if input folder is provided
if [ -z "$input_folder" ]; then
  echo "Error: Please provide the input folder."
  exit 1
fi

# Check if output file name is provided
if [ -z "$output_file" ]; then
  echo "Error: Please provide the output file name."
  exit 1
fi

# Check if input folder exists
if [ ! -d "$input_folder" ]; then
  echo "Error: Input folder does not exist."
  exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
  echo "Error: ffmpeg is not installed. Please install ffmpeg and try again."
  exit 1
fi

# Run ffmpeg command to convert images to video
ffmpeg -framerate 30 -pattern_type glob -i "$input_folder/*.png" -c:v libx264 -pix_fmt yuv420p "/tmp/ffmpeg_tmp.mp4"

ffmpeg -i "/tmp/ffmpeg_tmp.mp4" "$output_file"

rm /tmp/ffmpeg_tmp.mp4

echo "Conversion completed successfully."

