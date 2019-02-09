# # -*- coding: utf8 -*-
# import tensorflow as tf
# import numpy as np
# import os
# import json
# import glob
# import re
# import sys
# import time
# import gensim
# import torch
# from tensorboardX import SummaryWriter
# from tensorflow.contrib.tensorboard.plugins import projector
# import codecs
# from pykakasi import kakasi
#
# kakasi = kakasi()
#
# kakasi.setMode('H', 'a')
# kakasi.setMode('K', 'a')
# kakasi.setMode('J', 'a')
#
# conv = kakasi.getConverter()
#
#
#
# def int_to_onehot(indices, depth):
#     return tf.reshape(tf.transpose(tf.one_hot(indices, depth)), [depth, 1])
#
#
# def numerical_sort(value):
#     numbers = re.compile(r'(\d+)')
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts
#
#
# def read_json_to_list(folder_path):
#     text_list = []
#     files = sorted(glob.glob(folder_path), key=numerical_sort)
#     for file in files:
#         with open(file, 'r') as f:
#             f = f.read()
#             dict = json.loads(f, "utf-8")
#             for key, value in sorted(dict.items(), key=lambda x: x[1]):
#                 if key == "":
#                     pass
#                 else:
#                     text_list.append(key)
#                     if value > 200:
#                         break
#     return text_list
#
#
# def read_json_to_dict(folder_path):
#     text_and_id = {}
#     files = sorted(glob.glob(folder_path), key=numerical_sort)
#     for file in files:
#         with open(file, 'r') as f:
#             dict = json.load(f)
#             text_and_id.update(dict)
#     return text_and_id
#
#
# def dict_value_to_onehot(dict):
#     dicts = {}
#     last_value = sorted(dict.values())[-1]
#     for key, value in sorted(dict.items(), key=lambda x: x[1]):
#         if value == 1:
#             pass
#         else:
#             dicts[key] = int_to_onehot(value-1, 200)
#             if value > 200:
#                 break
#         print(key)
#     return dicts
#
#
# texts = read_json_to_list("../test/*")
# print("coverted list!")
#
# dict = read_json_to_dict("../test/*")
# print("converted dict!")
# text_ids = dict_value_to_onehot(dict)
# print("converted onehot!")
#
#
# # for key, value in text_ids.items():
# #     print(key)
# #     with tf.Session() as sess:
# #         print(sess.run(value))
#
#
#
# w = tf.Variable(tf.random_normal([200, 200], -0.1, 0.1))
#
# x1 = tf.placeholder(tf.float32, [200, 1])
#
# x2 = tf.placeholder(tf.float32, [200, 1])
#
# b = tf.Variable(tf.random_normal([200, 1], -0.1, 0.1))
#
# y = tf.placeholder(tf.float32, [200, 1])
#
# y_ = tf.nn.softmax(tf.matmul(w, x1) + b + tf.matmul(w, x2), axis=2)
#
# init = tf.global_variables_initializer()
#
# loss = tf.reduce_mean(tf.square(y - y_))
#
# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(20):
#         for i in range(len(texts) - 2):
#             sess.run(train, feed_dict={
#                 x1: sess.run(text_ids[texts[i]]),
#                 x2: sess.run(text_ids[texts[i+2]]),
#                 y: sess.run(text_ids[texts[i+1]])
#             })
#             print(sess.run(loss, feed_dict={
#                 x1: sess.run(text_ids[texts[i]]),
#                 x2: sess.run(text_ids[texts[i+2]]),
#                 y: sess.run(text_ids[texts[i+1]])
#             }))
#     logdir = "./corpus/log"
#     saver = tf.train.Saver()
#     saver.save(sess, logdir + "/blog.ckpt", i)
#     summary_writer = tf.summary.FileWriter(logdir)
#     config = projector.ProjectorConfig()
#     embedding = config.embeddings.add()
#     embedding.tensor_name = w.name  # 右辺のembeddingsはこの関数の最初で定義したtf.Variable
#     embedding.metadata_path = "/Users/kandasora/Desktop/Python/word2vec/corpus/model/blog.metadata.tsv"
#     projector.visualize_embeddings(summary_writer, config)
#     words = "Index\tLabel\n"
#     i = 0
#     for key, value in text_ids.items():
#         print(key)
#         words += str(i) + "\t" + conv.do(key) + "\n"
#         i += 1
#     with open("./corpus/model/blog.metadata.tsv", "w") as f:
#         f.writelines(words.encode("utf-8"))
#     print("Embeddings metadata was saved to ./corpus/model/blog.metadata.tsv")

from gensim.models import word2vec
import codecs

with codecs.open("mecab_output.txt", 'r', 'utf-8') as f:
    corpus = f.read().splitlines()

corpus = [sentence.split() for sentence in corpus]

model = word2vec.Word2Vec(size=300, alpha=0.025, min_count=20, window=10, iter=1, compute_loss=True)
model.build_vocab(corpus)
for i in range(2000):
    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
    print(model.get_latest_training_loss())
model.save('model.model')
#
#
# model = word2vec.Word2Vec(size=300, min_count=1, window=10, iter=1000, compute_loss=True)
# model.build_vocab(corpus)
# model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)
# print(model.get_latest_training_loss())
#
# model.save('model.model')
#
# word_vector = model.wv.vocab.keys()
# print(len(word_vector))
