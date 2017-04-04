# TensorBoxPy3 https://github.com/SMH17/TensorBoxPy3

#Extract frames from video file saving them as .jpg pictures

import sys
import cv2

print(cv2.__version__)
if len(sys.argv) < 2:
    print("Wrong param, try something like:  yourvideofile.mp4")
    exit()
else:
	videofile=sys.argv[1]
	print("The input video is: ", videofile)
	vidcap = cv2.VideoCapture(videofile)
	success,image = vidcap.read()
	count = 0
	success = True
	print("Extraction in progress...")
	while success:
		success,image = vidcap.read()
		cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
		count += 1
	print("Completed!\nNumber of frames extracted: ", count)