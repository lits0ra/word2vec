import tensorflow as tf
import numpy as np
import os
import glob
import json


def onehot(start_num, finish_num):
    return tf.reshape(tf.transpose(tf.one_hot(start_num, finish_num)), [finish_num, 1])

with open("mecab_output.txt", "r") as texts:
    text_list = []
    for line in texts:
        line_split = line.split(" ")
        text_list.append(line_split)
value_num = 0
text_ids = {}

output_files = glob.glob("./outputs/*")
file_count = len(output_files)
for output_file in output_files:
    f = open(output_file, 'r')
    dict = json.load(f)

    text_ids.update(json.load(f))
f.close()

w = tf.Variable(tf.random_normal([10, 10], -0.01, 0.01))

x1 = tf.placeholder(tf.float32, [10, 1])

x2 = tf.placeholder(tf.float32, [10, 1])

b = tf.Variable(tf.random_normal([10, 1], -0.01, 0.01))

y = tf.placeholder(tf.float32, [10, 1])

y_ = tf.nn.softmax(tf.matmul(w, x1) + b + tf.matmul(w, x2), axis=2)

init = tf.global_variables_initializer()

loss = tf.reduce_mean(tf.square(y - y_))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
print(1)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(loss , feed_dict={
        x1: sess.run(text_ids[texts[1]]),
        x2: sess.run(text_ids[texts[3]]), 
        y: sess.run(text_ids[texts[2]])
    }))
    print("-----")
    for i in range(200):
        for i in range(len(texts) - 2):
            sess.run(train, feed_dict={
                x1: sess.run(text_ids[texts[i]]),
                x2: sess.run(text_ids[texts[i+2]]), 
                y: sess.run(text_ids[texts[i+1]])
            })
            print(sess.run(loss, feed_dict={
                x1: sess.run(text_ids[texts[i]]),
                x2: sess.run(text_ids[texts[i+2]]), 
                y: sess.run(text_ids[texts[i+1]])
            }))
            
    print(sess.run(w))
