import tensorflow as tf
import numpy as np

texts = ["I", "love", "to", "read", "books", "so", "I", "decided", "to", "go", "to", "a", "bookstore"]

value_num = 0

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



[-0.05690343 -0.00450222 -0.19951707 -0.13202114 -0.02692418 -0.13647765
  -0.0718652  -0.12443346 -0.1915391  -0.12478586]
 [-0.09893557 -0.11849919 -0.15204813 -0.07784621 -0.03142556 -0.29339615
   0.12719946 -0.09375459 -0.05861934 -0.04811548]
 [-0.00490104 -0.14968747 -0.01896276  0.11725914 -0.18253216 -0.11752768
  -0.228148   -0.09575126 -0.20878617 -0.04569921]
 [-0.24158005 -0.11574891 -0.03036129 -0.3238816  -0.15409178 -0.04049464
  -0.03059266 -0.14539567 -0.32818848  0.02915662]
 [-0.20004608 -0.09026866 -0.02843061 -0.08969345 -0.19865358 -0.1558164
  -0.18876208  0.01774561 -0.1760881  -0.14342676]
 [ 0.01050039 -0.02861719 -0.03277495 -0.14557078 -0.18394788  0.004941
  -0.14427781 -0.32681492  0.02764576  0.01274604]
 [-0.02301048 -0.17069718 -0.09648181 -0.09016206 -0.11850543 -0.05326793
  -0.03756822 -0.04362813 -0.07762832 -0.19391267]
 [ 0.03606458 -0.01291979 -0.10302477 -0.0352557   0.08253486 -0.2797869
  -0.17985985 -0.17524818 -0.02630109 -0.01886939]
 [-0.10614198 -0.01446237 -0.03765671 -0.0568916  -0.27868316 -0.04359926
  -0.08500551 -0.05079242  0.00316091  0.18691674]
 [-0.08178788 -0.13093276 -0.26299596 -0.07075518 -0.25172195 -0.11265464
  -0.05043673 -0.11327458 -0.02731059 -0.03396014]