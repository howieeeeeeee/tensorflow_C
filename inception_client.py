import tensorflow as tf
import numpy as np
import os
import time
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('graph_file', '/xhome/tx_youhuo/basic/graph/affective_frozen_graph.pb', "")
tf.app.flags.DEFINE_string('label_file', '/xhome/tx_youhuo/basic/Image/classes.txt', "")

def load_graph(graph_file):
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name = "")
    return graph

def getResult(image_file):
    graph = load_graph(FLAGS.graph_file)
    input_tensor = graph.get_tensor_by_name('input_node:0')
    output_tensor = graph.get_tensor_by_name('inception_v3/logits/predictions:0')
    
    with open(FLAGS.label_file, 'r+') as f:
        labels  = f.readlines()
    
    with tf.Session(graph = graph) as sess:
        start = time.time()
        result = sess.run(output_tensor, feed_dict = {input_tensor : image_file})
        end = time.time()
        
        print end-start
        scores = list(result[0])
        del scores[0]
        scores.sort()
        scores =  map(lambda x : round(x, 6), scores)

        result_index = result.argsort()[0]
        result_index = list(result_index)
        print result_index
        result_index.remove(0)
        classes = []
        for index in result_index:
            classes.append(labels[index - 1].strip())
        print classes
    
    emotions_result = {'classes' : classes, 'scores' : scores}
    return emotions_result

def detect(img_dir):
    face_cascade = cv2.CascadeClassifier('/xhome/tx_youhuo/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')

    imgData = cv2.imread(img_dir)

    #min_h = int(max(imgData.shape[0] / 20, 50))
    #min_w = int(max(imgData.shape[1] / 20, 50))
    gray = cv2.cvtColor(imgData, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors = 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(imgData, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    print faces
    cv2.imwrite('2.jpg', imgData)
    img = cv2.imread(img_dir)
    imgout = img[faces[0][1] : faces[0][1] + faces[0][3], faces[0][0] : faces[0][0] + faces[0][2], :]
    cv2.imwrite('imgout.jpg', imgout)
    
if __name__ == '__main__':
    result = getResult('1.jpg')
