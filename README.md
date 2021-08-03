# Face-Mask-Detection
This repo contains code of face mask detection on image and video.There are two models used in detecting face mask. FaceNet model detects face pixels in image or frame and MaskNet detects face mask in the detected faces.MaskNet is a pre-trained MobileNetv2 model that is further trained on [Face Mask Dataset](https://www.kaggle.com/omkargurav/face-mask-dataset "Kaggle Dataset").

## Installation 
Clone this repo

`git clone https://github.com/mazhar18941/Face-Mask-Detection.git`

Install packages

`pip3 install -r requirements.txt`

Face mask detection in images

`python3 image_inference.py -i <image_path>`

Face mask detection in video

`python3 video_inference.py`
