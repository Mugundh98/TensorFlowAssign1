#!/usr/bin/env python

import pandas as pd
import tensorflow as tf 
import numpy
import pandas as pd
import numpy as np
import scipy.io

import PIL.Image
import sys
import os
import numpy
import csv
import scipy.io
import cv2
all_labels = scipy.io.loadmat('/imagelabels.mat')['labels'][0] - 1

labels=pd.DataFrame(data=all_labels,index=None)
labels.columns=['labels']

mypath='/trail/jpg'
onlyfiles = [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]
imagesA = numpy.empty(len(onlyfiles), dtype=object)
#imag_binary = numpy.empty(len(onlyfiles), dtype=object)
img_binaryA=[]
for n in range(0, len(onlyfiles)):
    imagesA[n] = cv2.imread(os.path.join(mypath,onlyfiles[n]),cv2.IMREAD_GRAYSCALE)
    thresh = 128
    img_binA=(cv2.threshold(imagesA[n], thresh, 255, cv2.THRESH_BINARY)[1])
    width= 30
    height=30
    dim = (width, height)
    resized=cv2.resize(img_binA, dim, interpolation = cv2.INTER_AREA)
    img_binaryA.append(resized)

arrayA=[]

for i in range(0,len(img_binaryA)):
    arrayA.append(img_binaryA[i].reshape(1,900))

for j in range(0,200):
    for i in range(900):
        if(arrayA[j][0][i]<128):
            arrayA[j][0][i]=0
        else:
            arrayA[j][0][i]=1
resultFile1 = open('testing.csv','w+')
wr = csv.writer(resultFile1)
for i in range(0,len(arrayA)):
    wr.writerows(arrayA[i])

df_test= pd.read_csv("testing.csv",header=None,index_col=None)

df_label = pd.get_dummies(labels,columns=["labels"]).values
test_labels = all_labels[0:200]

image_size = 30
num_labels = 102
num_channels = 1 # grayscale
df_test1 = df_test.to_numpy()
df_test2 = df_test1.reshape((-1, image_size, image_size, num_channels))
df_test3 = df_test2.astype(np.float32)
df_test=df_test3

submission_test = np.pad(df_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')     

def LeNet_5(x):
   
  # Layer 1 : Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
  conv1_w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = 0.1))
  conv1_b = tf.Variable(tf.zeros(6))
  conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 

  conv1 = tf.nn.relu(conv1)

  # Pooling Layer. Input = 28x28x1. Output = 14x14x6.
  pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
  
  

  conv2_w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = 0.1))
  conv2_b = tf.Variable(tf.zeros(16))
  conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
  
  conv2 = tf.nn.relu(conv2)

  
  pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
  
  
  fc1 = tf.compat.v1.layers.flatten(pool_2)
    
  
  
  fc1_w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape = (400,120), mean = 0, stddev = 0.1))
  fc1_b = tf.Variable(tf.zeros(120))
  fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
  
  fc1 = tf.nn.relu(fc1)
    
  fc2_w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape = (120,84), mean = 0, stddev = 0.1))
  fc2_b = tf.Variable(tf.zeros(84))
  fc2 = tf.matmul(fc1,fc2_w) + fc2_b
  

  fc2 = tf.nn.relu(fc2)
    
  fc3_w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(shape = (84,102), mean = 0 , stddev = 0.1))
  fc3_b = tf.Variable(tf.zeros(102))
  logits = tf.matmul(fc2, fc3_w) + fc3_b
  return logits

tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32, shape=[None,34,34,1])
y_ = tf.compat.v1.placeholder(tf.int32, (None))

logits = LeNet_5(x)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = logits)

loss_operation = tf.math.reduce_mean(cross_entropy)

training_operation = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.compat.v1.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        # print(batch_y)
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.compat.v1.Session() as sess:
    # Restore variables from disk.
  saver = tf.compat.v1.train.Saver()
  saver.restore(sess, "/newtf/lenet.ckpt")
  print("Model restored.")
  Z = logits.eval(feed_dict={x: submission_test})
  y_pred = np.argmax(Z, axis=1)



count=0
for i in range(200):
  if(y_pred[i]==test_labels[i]):
    count=count+1
acc = count/y_pred.shape[0]
print("accuracy is",acc)

