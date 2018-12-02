import tensorflow as tf
import numpy as np

texts = ["I", "love", "to", "read", "books", "so", "I", "decided", "to", "go", "to", "a", "bookstore"]

value_num = 0

text_ids = {}

for key in texts:
    if key not in text_ids.keys():
        text_ids[key] = tf.reshape(tf.transpose(tf.one_hot(value_num, 10)), [10, 1])
        value_num = value_num + 1
print(text_ids)

w = tf.Variable(tf.random_normal([10, 10], -0.01, 0.01))

x1 = tf.placeholder(tf.float32, [10, 1])

x2 = tf.placeholder(tf.float32, [10, 1])

b = tf.Variable(tf.random_normal([10, 1], -0.01, 0.01))

y = tf.placeholder(tf.float32, [10, 1])

y_ = tf.nn.softmax(tf.matmul(w, x1) + b + tf.matmul(w, x2), axis=2)

init = tf.global_variables_initializer()

loss = tf.reduce_mean(tf.square(y - y_))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
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
