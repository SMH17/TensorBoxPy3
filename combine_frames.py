# TensorBoxPy3 https://github.com/SMH17/TensorBoxPy3

#Combine frame images into a video.
#e.g. python3 combine_frames.py -ext jpg -o output.mp4
#will combine all .jpg image files in a video named output.mp4

import cv2
import argparse
import os
import re

print("# TensorBoxPy3: combining frames in a video")

# Map the numeric part to allow numerical sort
def numericalSortMap(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())

# Arguments
dir_path = '.'
ext = args['extension']
output = args['output']

images = []
for f in os.listdir(dir_path):
    if f.endswith(ext):
        images.append(f)

# Determine the width and height from the first image
image_path = os.path.join(dir_path, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

print("Video creation in progress...")
count=0
for image in sorted(images, key=numericalSortMap):

    image_path = os.path.join(dir_path, image)
    frame = cv2.imread(image_path)

    if frame is None:
        print("Skipped frame: ",count)
    else:
        out.write(frame) # Write out frame to video
        cv2.imshow('video',frame)
        count+=1
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()

print("Total frames combined: ",count)
print("The output video is {}".format(output))
