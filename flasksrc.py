from flask import Flask, request, redirect, url_for, flash, render_template
#from tqdm import tqdm
import tensorflow as tf
import sys
import os
import cv2
import numpy as np
import glob
import argparse
import winsound
from utils import label_map_util
from utils import visualization_utils as vis_util
import shutil

UPLOAD_FOLDER = 'E:\\Projects\\ObjectDetection\\testvideo\\acc3.mp4'
ALLOWED_EXTENSIONS = set(['mp4'])

app = Flask(__name__)

def upload_video():
	shutil.copyfile('C:\\TensorflowModels\\models\\research\\object_detection\\acc6.mp4', UPLOAD_FOLDER)
	return render_template('runmodel.html')

def upload_file():
	#if request.method == 'POST':
	print('zero')
	if 'file' not in request.files:
		flash('No file part')
		print('one')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No selected file')
		print('two')
		return redirect(request.url)
	if file and allowed_file (file.filename):
    		filename = secure_filename(file.filename)
    		print('three')
    		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	return render_template('runmodel.html')

def accident_detection():
	print("car crash detection")
	MODEL_NAME = 'inference_graph'

	CWD_PATH = os.getcwd()

	PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

	PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

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

	count = 1

	cap = cv2.VideoCapture('C:\\TensorflowModels\\models\\research\\object_detection\\test_video\\acc6.mp4')
	while (cap.isOpened()):
		ret, frame = cap.read()
		image_expanded = np.expand_dims(frame, axis=0)
		(boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})
		# if args.textual:
		print("\n\n"+"="*50+"	Results    "+"="*50+"\n\n")
		print("        Class               Surety")
		print()
		count = 0
		for i in range(scores.shape[1]):
			if scores[0,i]>0.8:
				print("    "+str(i+1)+".  "+str(category_index.get(classes[0,i])['name'])+"    ==>    "+str(scores[0,i]*100)+' %')
				print()
				count+=1
				
				if(str(category_index.get(classes[0,i])['name'])=="crashed"):
					winsound.Beep(440, 500)
		print("\n	Total "+str(count)+" objects classified.\n")


		vis_util.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8,min_score_thresh=0.80)
		cv2.imshow('Object detector', frame)
		count+=1

		cv2.waitKey(1)

	cap.release()

	cv2.destroyAllWindows()


@app.route("/")
def home():
	return render_template('base_rc.html')

@app.route('/runmodel')
def runmodel():
	accident_detection()
	#upload_file()
	return render_template('runmodel.html')
 
if __name__ == "__main__":
	app.secret_key='super secret key'
	app.run('127.0.0.1')