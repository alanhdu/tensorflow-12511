import numpy as np
import tensorflow as tf

SIZE = 96

data = np.load("final.npz")
y_train = data["y"]
X_train = data["X"]

truth = tf.placeholder(tf.int32, [None, None], "truth")
inputs = tf.placeholder(tf.float32, [None, None, 97], "input")

x = tf.layers.dense(inputs, SIZE, activation=tf.nn.relu)

lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(SIZE)
output, __ = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32)
logits = tf.layers.dense(output, 98)

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=truth)
loss = tf.reduce_mean(loss)

train_step = tf.contrib.layers.optimize_loss(
    loss, None, 1e-3, tf.train.AdamOptimizer, clip_gradients=5.0, summaries=[])

saver = tf.train.Saver(tf.trainable_variables())

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, "weights/model-0")

feed_dict = {
    inputs: X_train,
    truth: y_train,
}

print([np.isfinite(v).all() for v in sess.run(tf.trainable_variables())])
sess.run(train_step, feed_dict)
print([np.isfinite(v).all() for v in sess.run(tf.trainable_variables())])
