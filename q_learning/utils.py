import os
import tensorflow as tf

def load_checkpoint(saver, dir, sess):
    ckpt = tf.train.get_checkpoint_state(dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Restored from checkpoint {}".format(ckpt.model_checkpoint_path))
    else:
        print("No checkpoint")

def save_checkpoint(saver, dir, sess, step=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_path = saver.save(sess, dir + '/graph', step)
    print("Models saved in file: {} ...".format(save_path))
