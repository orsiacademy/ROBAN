# Intro?

# Usage
## Offline
To execute ROBAN offline on a local video, offline_inference.py can be run from command line as follows:

```python offline_inference.py <input/video/path> <output/directory/path>```

Running this command will save a .JSON-file containing the predicted labels in the specified output directory. Optionally, --save_video can be added to the command and the resulting video, where out-of-body frames are blurred, will be saved in the output directory as well.
## Live
To execute ROBAN in a live setting and inferencing a live video stream, live_inference.py can be run from command line as follows:

```python live_inference.py```

The script will take the Webcam source (VideoCapture(0)) as input video, we used NDI (https://ndi.video/) to achieve this. It will display the video in an OpenCV window, which dimensions can be adjusted using the optional arguments output_width and output_height.

# Credits?
# License?
