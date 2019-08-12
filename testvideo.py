# Import packages
from tqdm import tqdm
import tensorflow as tf
import sys
import os
import cv2
import numpy as np
import glob
import argparse
import winsound

parser = argparse.ArgumentParser()
parser.add_argument('image_path', help="Absolute or relative path to an image.")
parser.add_argument('--with_image', help="Present results on image", action="store_true")
parser.add_argument('--textual', help="Present results in textual format", action="store_true")

args = parser.parse_args()

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

PATH_TO_IMAGE = args.image_path

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    	od_graph_def = tf.GraphDef()
    	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        	serialized_graph = fid.read()
        	od_graph_def.ParseFromString(serialized_graph)
        	tf.import_graph_def(od_graph_def, name='')

    	sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

dir_path = 'C:\\TensorflowModels\\models\\research\\object_detection\\test'
files = os.listdir(dir_path)

count = 1

for file in files:
    	vid = os.path.join(dir_path, file)
    	cap = cv2.VideoCapture(vid)
    	while (cap.isOpened()):
    		ret, frame = cap.read()
    		image_expanded = np.expand_dims(frame, axis=0)
    		(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})
    		vis_util.visualize_boxes_and_labels_on_image_array(frame,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8,min_score_thresh=0.80)
    		cv2.imshow('Object detector', frame)
    		cv2.imwrite("detectedimage%d.jpg" % count, frame)
    		count+=1
    		cv2.waitKey(1)
    	cap.release()
cv2.destroyAllWindows()