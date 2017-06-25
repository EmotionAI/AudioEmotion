for file in *.mp4; do ffmpeg -i "$file" "${file%.mp4}".wav; done
