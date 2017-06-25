for file in *.mp4; 
do ffmpeg -y -i "$file" "${file%.mp4}".wav; 
done
