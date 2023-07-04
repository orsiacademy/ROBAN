import argparse
import json
import os

import cv2
import torch
import torch.nn as nn
from PIL import Image
from alive_progress import alive_bar
from torchvision import transforms


def main(args):
    VIDEO_PATH = args.video_path
    OUTPUT_DIR = args.output_dir
    MODEL_PATH = args.model_path
    FRAME_PERIOD = args.frame_period
    SAVE_VIDEO = args.save_video

    assert os.path.exists(VIDEO_PATH), f"Given path to the video does not exist, got: {VIDEO_PATH}"
    assert os.path.exists(MODEL_PATH), f"Given path to model PTH-file does not exist, got: {MODEL_PATH}"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Compose the transforms that should be applied to the frames to inference
    TF = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.96, 1), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the model and send to DEVICE
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL = torch.load(MODEL_PATH)
    MODEL.eval()
    MODEL.to(DEVICE)

    # Open the VideoCapture and retrieve some properties
    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(VIDEO_PATH).split('.')[0]

    # Setup the outputs: a dictionary for the labels and, if SAVE_VIDEO is flagged a VideoWriter
    res = {
        'labels': [],
        'frames': []
    }
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        output_video_name = video_name + '_anonymized.mp4'
        out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, output_video_name), fourcc, fps, (w, h))

    # Initialize the frame number at -1 and start while-loop (reading the video) using a progress bar.
    frame_number = -1
    running_pred = None
    with alive_bar(int(num_frames)) as bar:
        while True:
            frame_number += 1
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference on the frame every FRAME_PERIOD frames
            if (frame_number % FRAME_PERIOD) == 0:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_tf = TF(img)
                img_tf = img_tf.to(DEVICE)
                outputs = nn.Sigmoid()(torch.squeeze(MODEL.forward(img_tf[None, ...]), dim=1))
                pred = int(torch.gt(outputs, 0.5))
                running_pred = pred

            # Add the label and frame number to the dictionary
            res['labels'].append(running_pred)
            res['frames'].append(frame_number)

            # Construct the anonymized output frame if outside body is detected. Write output frame to VideoWriter
            if SAVE_VIDEO:
                if running_pred == 0:
                    output_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    temp = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_LINEAR)
                    output_frame = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                out.write(output_frame)
            bar()

    # Save the dictionary
    output_labels_name = video_name + '_labels.json'
    with open(os.path.join(OUTPUT_DIR, output_labels_name), 'w') as json_writer:
        json.dump(res, json_writer)

    cap.release()
    if SAVE_VIDEO:
        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the input video.", type=str)
    parser.add_argument("output_dir", help="Path to the output directory.", type=str, default=None)
    parser.add_argument("--model_path", help="Path to the model weights (PTH-file).", type=str, default='./ROBAN.pth')
    parser.add_argument("--frame_period", help="Period of frame inference: every nth frame is inferenced.", type=int,
                        default=1)
    parser.add_argument("--save_video", action='store_true', help="Indicate whether the video should be saved or not.")

    main(parser.parse_args())
