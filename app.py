import tensorflow as tf
import numpy as np


def onehot(num):
    return tf.reshape(tf.transpose(tf.one_hot(num, 10)), [10, 1])

text_ids = {"I": onehot(1),
            "love": onehot(2),
            "to": onehot(3),
            "read": onehot(4),
            "books": onehot(5),
            "so": onehot(6),
            "decided": onehot(7),
            "go": onehot(8),
            "a": onehot(9),
            "bookstore": onehot(10)
            }

# for key in texts:
#     if key not in text_ids.keys():
#         text_ids[key] = tf.reshape(tf.transpose(tf.one_hot(value_num, 10)), [10, 1])
#         value_num = value_num + 1
# print tf.Session().run(text_ids)

w = tf.Variable(tf.random_normal([10, 10], -0.01, 0.01))

x1 = tf.placeholder(tf.float32, [10, 1])

x2 = tf.placeholder(tf.float32, [10, 1])

b = tf.Variable(tf.random_normal([10, 1], -0.01, 0.01))

y = tf.placeholder(tf.float32, [10, 1])

y_ = tf.nn.softmax(tf.matmul(w, x1) + b + tf.matmul(w, x2), axis=2)

init = tf.global_variables_initializer()

loss = tf.reduce_mean(tf.square(y - y_))

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(text_ids))
    for i in range(100):
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
