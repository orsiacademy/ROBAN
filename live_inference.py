import argparse
import os

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def main(args):
    MODEL_PATH = args.model_path
    FRAME_PERIOD = args.frame_period
    OUTPUT_WIDTH = args.output_width
    OUTPUT_HEIGHT = args.output_height

    assert os.path.exists(MODEL_PATH), f"Given path to model PTH-file does not exist, got: {MODEL_PATH}"

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
    cap = cv2.VideoCapture(0)
    if OUTPUT_WIDTH is None:
        OUTPUT_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if OUTPUT_HEIGHT is None:
        OUTPUT_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the frame number at -1 and start while-loop (reading the video) using a progress bar.
    frame_number = -1
    running_pred = None
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

        # Construct the anonymized output frame if outside body is detected. Display the output frame.
        if running_pred == 0:
            output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
        else:
            temp = cv2.resize(frame, (16, 16), interpolation=cv2.INTER_LINEAR)
            output_frame = cv2.resize(temp, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('ANONYMIZED', output_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Path to the model weights (PTH-file).", type=str, default='./ROBAN.pth')
    parser.add_argument("--frame_period", help="Period of frame inference: every nth frame is inferenced.", type=int,
                        default=4)
    parser.add_argument("--output_width", help="Width of the window displaying the output.", type=int, default=None)
    parser.add_argument("--output_height", help="Height of the window displaying the output.", type=int, default=None)

    main(parser.parse_args())
