import os
from flask import Flask, render_template, request
from werkzeug import secure_filename
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image
import logging
import time
import pandas as pd
import pickle

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

log_file_path = dir_path + "\\LogFile.log"
logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
logger = logging.getLogger()

from utils import label_map_util
from utils import visualization_utils as vis_util
MODEL_NAME = 'Sport_Detection_Graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'obj_detect.pbtxt')
NUM_CLASSES = 20

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

sport_classifier_file = 'data/sport_detect_model.sav'
sport_classifier = pickle.load(open(sport_classifier_file, 'rb'))
sports_dict = {0:'Cricket', 1:'Football', 2:'Tennis'}

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


app = Flask(__name__)

app.config['TEST_FOLDER'] = 'test_images/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['TEST_FOLDER'], filename))
    
    PATH_TO_TEST_IMAGES_DIR = app.config['TEST_FOLDER']
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,filename.format(i)) for i in range(1, 2) ]
    IMAGE_SIZE = (12, 8)
    image_number = 1
    df = pd.DataFrame(columns=range(1,21))
	
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                uploaded = time.clock()
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (scores, classes, num_detections) = sess.run([scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
                scores = scores[0]
                classes = classes[0].astype(np.uint8)

        confident_detections = []
        list_to_add_to_df = []
        
        for i in range(0,int(num_detections[0])):
            if scores[i] >0.9:
                confident_detections.append(classes[i])
        for j in df.columns:
            list_to_add_to_df.append(confident_detections.count(j))
            
        df.loc[image_number] = list_to_add_to_df
        output_class = sport_classifier.predict(df)
        output_class = np.where(output_class == 1)[1]
        output_class = [sports_dict[key] for key in output_class]

        testing = time.clock() - uploaded
    return " Detected Sport is " + output_class[0] + ". Time taken for detection: " + str(round(testing, 3)) + " secs"

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)

