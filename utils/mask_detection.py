import tensorflow as tf

def load_model(path):
    return tf.keras.models.load_model(path)

def mask_detection(mask_detector, resized_image):
    pred = mask_detector.predict(resized_image)
    #print(pred)
    if pred < 0.35:
        return True
    else:
        return False