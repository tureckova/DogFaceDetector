import keras
import cv2
import numpy as np
from utils import decode_netout, draw_boxes

IMAGE_PATH = 'puppies.jpg'
MODEL_PATH = 'tiny_yolo_entire_model.h5'
ANCHORS = [2.17,2.64, 3.55,3.91, 5.01,6.63, 6.20,4.60, 8.59,8.49]
LABELS = ["dog"]
MAX_BOX_PER_IMAGE = 10

# load model
model = keras.models.load_model(MODEL_PATH)

# load and prepare image
image = cv2.imread(IMAGE_PATH)
input_image = cv2.resize(image, (300, 300))
input_image = input_image / 255.
input_image = input_image[:,:,::-1]
input_image = np.expand_dims(input_image, 0)
dummy_array = np.zeros((1,1,1,1,MAX_BOX_PER_IMAGE,4))

# get predictions
netout = model.predict([input_image, dummy_array])[0]
boxes = decode_netout(netout, ANCHORS, len(LABELS), obj_threshold=0.5, nms_threshold=0.5)

# draw the boxes into original image
image = draw_boxes(image, boxes, LABELS)
cv2.imshow('Output', image);
cv2.waitKey();
cv2.destroyAllWindows()