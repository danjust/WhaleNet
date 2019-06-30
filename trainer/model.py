"""API to create a siamese net model"""


import tensorflow as tf
import numpy as np
import os
from trainer import convnets


class whalenet():
    """Model class"""
    def __init__(self, iterator, learning_rate, model_name, margin,
                 convnet=convnets.WhaleCNN):
        self.iterator = iterator
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.convnet = convnet
        self.margin = margin

        self._loss = None
        self._train_op = None
        self._summary_op = None

        self.global_step = tf.get_variable('global_step',
                                           initializer=tf.constant(0),
                                           trainable=False)


    @property
    def loss(self):
        """Funtion creates two identical CNNs,
        computes the difference of output features
        and returns the contrastive loss
        """
        if self._loss is None:
            next_batch = self.iterator.get_next()
            img1, img2, target_diff = next_batch

            features1 = self.convnet(img1, reuse=False)
            features2 = self.convnet(img2, reuse=True)
            results_diff = tf.sqrt(tf.reduce_mean(
                tf.square(features1 - features2),
                axis=1))
            results_diff = tf.reshape(results_diff,[-1,1])

            self._loss = (
                tf.reduce_mean((1-target_diff)*results_diff**2 / 2
                + target_diff*(tf.maximum(0.,self.margin-results_diff))**2/2,axis=1))
        return self._loss

    @property
    def train_op(self):
        """Function to create training operation with Adam optimizer"""
        if self._train_op is None:
            opt = tf.train.AdamOptimizer(
                learning_rate = self.learning_rate)
            self._train_op = opt.minimize(self.loss,
                                          global_step=self.global_step)
        return self._train_op

    @property
    def summary_op(self):
        """Function to write the summary, returns property"""
        if self._summary_op is None:
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self._summary_op =  tf.summary.merge_all()
        return self._summary_op


    def train(self, train_steps=5000, print_step=500):
        """Trains model and saves to checkpoints
        arguments:
            train_steps: number of training steps (int)
            print_step: output/summary after this number of steps (int)
        """
        _train_op = self.train_op               # build the model
        print("building model")

        init = tf.global_variables_initializer()

        try:
            os.mkdir('./checkpoints/%s' %self.model_name)
        except:
            pass

        saver = tf.train.Saver()
        l = np.zeros(train_steps)
        with tf.Session() as sess:
            sess.run(self.iterator.initializer)
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                './checkpoints/%s/checkpoint' % self.model_name))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored, step = %d" % self.global_step.eval())

            writer_train = tf.summary.FileWriter(
                './graphs/prediction/train/%s'
                %self.model_name, sess.graph)
            initial_step = self.global_step.eval()

            avg_loss = 0
            for step in range(initial_step, initial_step+train_steps):
                _,loss = sess.run(
                    [_train_op,self.loss])#,self.summary_op])
                avg_loss = avg_loss+np.mean(loss)/print_step
                if ((step+1)%print_step==0):
                    # writer_train.add_summary(summary, global_step=step)
                    print('Step {}: Train loss {:.3f}'.format(step, avg_loss))
                    avg_loss = 0
            writer_train.close()

            saver.save(sess, './checkpoints/%s/training' % self.model_name, step)
