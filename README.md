# Privacy-proof live surgery streaming: Development and validation of a low-cost, real-time robotic surgery anonymization algorithm

Background: Surgical video sharing has become common practice and requires complete anonymization when videos are shared outside of clinical trials. In case of endoscopic surgery, anonymization also requires removal or blurring of all video frames where the endoscope is outside of the body as it can record the patient or operating room staff. No non-commercial real-time algorithm exists which anonymizes live surgery streaming and no previous studies have tested anonymization across different robotic platforms.
Methods: A dataset of 63 surgical videos of 6 different procedures performed on 4 different surgical robotic systems is sampled and manually labelled as inside or outside the body. A deep learning model is trained on this dataset of 496.828 images to automatically detect out-of-body frames. Our solution is subsequently benchmarked to existing laparoscopic anonymization solutions using comparable annotation methods. We add an additional post-processing method to further boost performance in the offline setting as well as enable and test a low-cost setup for real-time anonymization during live surgery streaming.
Results: Framewise anonymization results in a ROC AUC-score of 99.46% on the unseen procedures in the test set. Our Robotic Anonymization Network (ROBAN) outperforms previous state-of-the-art algorithms and works in real-time. Further offline post-processing increases the ROC AUC-score to 99.61%. ROBAN also outperforms the state-of-the-art solutions on laparoscopic cholecystectomy (LCE), even though it is not trained on LCE whilst existing state-of-the-art solutions were.
Conclusions: Our deep learning model allows for robust anonymization of robotic surgery streaming, across robotic platforms, as well as post-hoc procedural anonymization.


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
