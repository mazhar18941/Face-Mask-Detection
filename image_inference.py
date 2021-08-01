import cv2
import  argparse
from utils import image, mask_detection, face_detection

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	default="images/image.jpeg",
	help="path to input image")
ap.add_argument("-fm", "--face_model", type=str,
	default="models/deploy.prototxt.txt",
	help="path to face detector model")
ap.add_argument("-fw", "--face_weights", type=str,
	default="models/res10_300x300_ssd_iter_140000.caffemodel",
	help="path to face detector model weights")
ap.add_argument("-md", "--mask_model", type=str,
	default="models/face_mask_model.h5",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


caffeModel = args['face_model']
prototextPath = args['face_weights']
mask_model_path = args['mask_model']
image_path = args['image']
threshold = args['confidence']

face_detector = face_detection.load_model(prototextPath, caffeModel)
mask_detector = mask_detection.load_model(mask_model_path)


img = image.load_image(image_path)

detections = face_detection.get_detection(face_detector, img)
bb_coord_pixel = face_detection.get_face_pixels(img, detections, threshold=threshold)

new_size = mask_detector.layers[0].input_shape[0][1:3]


def get_bb_label(mask_detector, bb_coord_pixel):
    bb_dic = {'bb_coord': bb_coord_pixel[0], 'bb_pixel': bb_coord_pixel[1],
              'bb_label': []}

    for bb_pixel in bb_dic['bb_pixel']:

        resized_img = image.resize_image(bb_pixel, new_size=new_size)
        mask = mask_detection.mask_detection(mask_detector, resized_img)
        if mask:
            # print(mask_detector(resized_img)[1])
            bb_dic['bb_label'].append('Mask')
        else:
            # print(mask_detector(resized_img)[1])
            bb_dic['bb_label'].append('No Mask')

    return bb_dic


coord_pixels_label = get_bb_label(mask_detector, bb_coord_pixel)

if coord_pixels_label:
    for coord, label in zip(coord_pixels_label['bb_coord'],
                            coord_pixels_label['bb_label']):
        (startX, startY, endX, endY) = coord
        text = label
        if text == 'Mask':
            clr = (255, 0, 0)
        else:
            clr = (0, 0, 255)

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY),
                      clr, 1)
        cv2.putText(img, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Output', img)
    print('***PRESS ANY KEY TO QUIT***')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print('Face was not detected in image')