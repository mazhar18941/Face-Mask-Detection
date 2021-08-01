import cv2
import numpy as np
#import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = plt.imread(image_path)
    return image

def resize_image(image, new_size):
    re_image = cv2.resize(image, new_size)
    re_image = np.expand_dims(re_image, axis=0)
    return re_image

def mask_detection(resized_image):
    pred = mask_detector.predict(resized_image)
    print(pred)
    if pred < 0.5:
        return True
    else:
        return False