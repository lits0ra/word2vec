# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
import os
import json
import glob
import re
import sys
import time


def int_to_onehot(indices, depth):
    return tf.reshape(tf.transpose(tf.one_hot(indices, depth)), [depth, 1])


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def read_json_to_list(folder_path):
    text_list = []
    files = sorted(glob.glob(folder_path), key=numerical_sort)
    for file in files:
        with open(file, 'r') as f:
            f = f.read()
            dict = json.loads(f, "utf-8")
            for key, value in sorted(dict.items(), key=lambda x: x[1]):
                if key == "":
                    pass
                else:
                    text_list.append(key)
                    if value > 10:
                        break
    return text_list


def read_json_to_dict(folder_path):
    text_and_id = {}
    files = sorted(glob.glob(folder_path), key=numerical_sort)
    for file in files:
        with open(file, 'r') as f:
            dict = json.load(f)
            text_and_id.update(dict)
    return text_and_id


def dict_value_to_onehot(dict):
    dicts = {}
    last_value = sorted(dict.values())[-1]
    for key, value in sorted(dict.items(), key=lambda x: x[1]):
        if value == 1:
            pass
        else:
            dicts[key] = int_to_onehot(value-2, 10)
            if value > 100:
                break
    return dicts


texts = read_json_to_list("./test/*")
print("coverted list!")
dict = read_json_to_dict("./test/*")
print("converted dict!")
text_ids = dict_value_to_onehot(dict)
# text_ids = json.dumps(str(text_ids), ensure_ascii=False, indent=2)
print("converted onehot!")


read_json_to_dict("./outputs/*")

w = tf.Variable(tf.random_normal([10, 10], -0.1, 0.1))

x1 = tf.placeholder(tf.float32, [10, 1])

x2 = tf.placeholder(tf.float32, [10, 1])

b = tf.Variable(tf.random_normal([10, 1], -0.1, 0.1))

y = tf.placeholder(tf.float32, [10, 1])

y_ = tf.nn.softmax(tf.matmul(w, x1) + b + tf.matmul(w, x2), axis=2)

init = tf.global_variables_initializer()

loss = tf.reduce_mean(tf.square(y - y_))

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# for i in range(1000000):
#     for i in range(len(texts) - 2):
#         print i
#
with tf.Session() as sess:
    sess.run(init)
    for i in range(1):
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