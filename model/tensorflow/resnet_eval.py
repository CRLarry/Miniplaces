import tensorflow as tf
import numpy as np
import scipy.misc
import os
import resnet as res
from DataLoader import *


# Dataset Parameters
batch_size = 50
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])
"""
# Training Parameters
learning_rate = 0.0001
dropout = 0.6 # Dropout, probability to keep units
training_iters = 20000
step_display = 50
step_save = 10000
"""
path_save = 'resnet18'
start_from = 'model/resnet18/resnet18v3-6000'
# Construct dataloader
data_test = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',
    'data_list': '../../data/test.txt',
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

#loader_test = DataLoaderDisk(**data_test)
"""
# tf Graph input
x = tf.placeholder(tf.float32, [1, fine_size, fine_size, c])
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

# construct model
logits = alexnet_bn(x, keep_dropout, train_phase) 
top_k = tf.nn.top_k(logits, k=5)

# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()
"""
x = tf.placeholder(tf.float32, [None,fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)
train_phase = tf.placeholder(tf.bool)

resnet = res.imagenet_resnet_v2(18,100)
logits = resnet(x,True)

saver = tf.train.Saver()
# launch the inference graph
with tf.Session() as sess:
    if len(start_from) > 1:
        saver.restore(sess,start_from)
    else:
        sess.run(init)
    # saver.restore(sess, tf.train.latest_checkpoint(restore_dir))
    #sess.run(init)
    
    top_k = tf.nn.top_k(tf.nn.softmax(logits),5)    

    #images = loader_test.get_test_images()
    #paths = loader_test.get_file_list()

    test_directory = '../../data/images/test'
    all_files = os.listdir(test_directory)
    
    class_size = int(len(all_files) / 100)
    with open("test.txt","w") as f:
        for i in range(100):
            images = []
            names = []
            for j in range(class_size * i, min(len(all_files), class_size*(i+1))):
                path = os.path.join(test_directory, all_files[j])
                test_image = scipy.misc.imread(path)
                test_image = scipy.misc.imresize(test_image,(load_size,load_size))
                test_image = test_image.astype(np.float32)/255.
                test_image -= np.array(data_mean)
                
                h = int((load_size-fine_size)/2)
                w = int((load_size-fine_size)/2)
                
                test = test_image[h:h+fine_size, w:w+fine_size, :]
                images.append(test)
                names.append(all_files[j])
            
            images = np.array(images)
            preds,indices = sess.run(top_k, feed_dict={x:images, keep_dropout: 1., train_phase:False})
            for i in range(len(names)):
                name = names[i]
                prediction = " ".join(map(str, indices[i]))
                f.write("test/" + name + " " + prediction + "\n")
    
print ("FINISHED EVALUATING TEST SET")
