"""API to create a siamese net model"""


import tensorflow as tf
import numpy as np
import os
from whaleid import convnets


class whalenet():
    """Model class"""
    def __init__(self, iterator, learning_rate, model_name,
                 convnet=convnets.WhaleCNN):
        self.iterator = iterator
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.convnet = convnet

        self.img1 = None
        self.img2 = None
        self.target_diff = None
        self._results_diff = None
        self._contrastive_loss = None
        self._train_op = None
        self._summary_op = None

        self.global_step = tf.get_variable('global_step',
                                           initializer=tf.constant(0),
                                           trainable=False)


    @property
    def results_diff(self):
        """Creates two identical CNNs and
        returns difference of output features
        """
        if self._results_diff is None:

            features1 = self.convnet(self.img1, reuse=False)
            features2 = self.convnet(self.img2, reuse=True)
            self._results_diff = tf.sqrt(tf.reduce_mean(
                tf.square(features1 - features2),
                axis=1))
            self._results_diff = tf.reshape(self._results_diff,[-1,1])
        return self._results_diff

    @property
    def contrastive_loss(self, margin=1):
        """Contrastive loss function for siamese networks"""
        if self._contrastive_loss is None:
            self._contrastive_loss = (
                tf.reduce_mean((1-self.target_diff)*self.results_diff**2 / 2
                + self.target_diff*(tf.maximum(0.,margin-self.results_diff))**2/2,axis=1))
        return self._contrastive_loss

    @property
    def train_op(self):
        """Function to create training operation with Adam optimizer"""
        if self._train_op is None:
            opt = tf.train.AdamOptimizer(
                learning_rate = self.learning_rate)
            self._train_op = opt.minimize(self.contrastive_loss,
                                          global_step=self.global_step)
        return self._train_op

    @property
    def summary_op(self):
        """Function to write the summary, returns property"""
        if self._summary_op is None:
            tf.summary.scalar("loss", self.contrastive_loss)
            tf.summary.histogram("histogram_loss", self.contrastive_loss)
            self._summary_op =  tf.summary.merge_all()
        return self._summary_op


    def train(self, trainsteps=5000, printstep=500):
        """Trains model and saves to checkpoints
        arguments:
            trainsteps: number of training steps (int)
            printstep: output/summary after this number of steps (int)
        """
        _train_op = self.train_op               # build the model

        try:
            os.mkdir('./checkpoints/%s' %self.model_name)
        except:
            pass

        saver = tf.train.Saver()
        l = np.zeros(trainsteps)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.iterator.initializer)
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(
                './checkpoints/%s/checkpoint' % self.model_name))
            # if that checkpoint exists, restore from checkpoint
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restored, step = %d" % self.global_step.eval())

            writer_train = tf.summary.FileWriter('./graphs/prediction/train/%s'
                %self.model_name, sess.graph)
            initial_step = self.global_step.eval()

            avg_loss = 0
            next_batch = self.iterator.get_next()
            for step in range(initial_step, initial_step+trainsteps):
                self.img1, self.img2, self.target_diff = next_batch

                # target, real, loss = sess.run([self.target_diff, self.results_diff, self.contrastive_loss])
                # print(target)
                # print(real)
                # print(loss)

                _,loss = sess.run(
                    [self.train_op,self.contrastive_loss])#,self.summary_op])
                avg_loss = avg_loss+np.mean(loss)/printstep
                if ((step+1)%printstep==0):
                    # writer_train.add_summary(summary, global_step=step)
                    print('Step {}: Train loss {:.3f}'.format(step, avg_loss))
                    avg_loss = 0
            writer_train.close()

            saver.save(sess, './checkpoints/%s/training' % self.model_name, step)
