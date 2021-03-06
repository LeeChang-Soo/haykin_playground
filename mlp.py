import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io
import math
from pprint import pprint as pp

import circle

def get_uniform_range(m):
    return math.sqrt(3/math.sqrt(m))

class circleClf(object):

    def set_learning_rate(self):
        self.starter_learning_rate = 1e-1
        self.end_learning_rate = 1e-5
        self.learning_rate = tf.train.polynomial_decay(
            self.starter_learning_rate,
            self.global_step,
            2000*self.max_epoch,
            end_learning_rate=self.end_learning_rate,
            power=1.0,
            cycle=False,
            name=None,
        )
        return

    def get_tag(self):
        return 'naive' + ('_normalized' if self.normalize else '_not_normalized')

    def act_fn(self, in_tensor):
        return tf.tanh(in_tensor)

    def initializer(self, shape, m):
        return tf.random_uniform(shape, minval=0., maxval=1.)

    def __init__(self, max_epoch=50, normalize=True):
        self.normalize = normalize
        with tf.name_scope(self.get_tag()) as scope:
            hidden = 20
            self.max_epoch = max_epoch
            self.global_step = tf.Variable(0, trainable=False)
            self.set_learning_rate()
            # Placeholder
            self.input_layer = tf.placeholder(tf.float32, [None, 2], name='input')
            self.label_layer = tf.placeholder(tf.int32, [None, 1], name='label')

            self.w1 = tf.Variable(self.initializer([2, hidden], 2))
            self.b1 = tf.Variable(self.initializer([hidden], 2))
            tf.summary.histogram('w1', self.w1)

            self.w2 = tf.Variable(self.initializer([hidden, 1], hidden))
            self.b2 = tf.Variable(self.initializer([1], hidden))
            tf.summary.histogram('w2', self.w2)

            # network construction
            self.hidden_layer_1_v = tf.matmul(self.input_layer, self.w1) + self.b1
            self.hidden_layer_1_y = self.act_fn(self.hidden_layer_1_v)

            self.output_layer_v = tf.matmul(self.hidden_layer_1_y, self.w2) + self.b2
            self.output_layer_y = tf.tanh(self.output_layer_v)

            self.loss = tf.losses.mean_squared_error(self.label_layer, self.output_layer_y)

            tf.summary.scalar("loss", self.loss)

            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

            self.plot_buf_ph = tf.placeholder(tf.string)
            image = tf.image.decode_png(self.plot_buf_ph, channels=4)
            image = tf.expand_dims(image, 0)  # make it batched
            self.plot_image_summary = tf.summary.image('fig', image, max_outputs=1)

        self.summary = tf.summary.merge_all()
        return

    def make_color_plot(self, sess, input_layer, output_layer):
        width = 0.5
        xs = np.arange(self.min_x, self.max_x, width)
        ys = np.arange(self.min_y, self.max_y, width)
        Xs, Ys = np.meshgrid(xs, ys)
        Zs = []
        for xs, ys in zip(Xs, Ys):
            zs = []
            for x, y in zip(xs, ys)[:-1]:
                if self.normalize:
                    datum = (np.array([x, y]) - self.avg) / self.std
                else:
                    datum = np.array([x, y])
                output_value = sess.run([output_layer], feed_dict={
                        self.input_layer : [datum],
                    })
                value = output_value[0][0]
                zs.append(1 if value >= 0 else -1)
            Zs.append(zs)
        Zs = np.reshape(np.array(Zs), [Xs.shape[0], Xs.shape[1]-1])
        #  pp(Xs.shape)
        #  pp(Ys.shape)
        #  pp(Zs.shape)
        return Xs, Ys, Zs

    def make_decision_line(self, sess, input_layer, output_layer):
        xs = np.arange(-15, 25, 1)
        ys = np.arange(-30, 30, 0.5)
        ys_res = []
        for x in xs:
            output_values = []
            for y in ys:
                if self.normalize:
                    datum = (np.array([x, y]) - self.avg) / self.std
                else:
                    datum = np.array([x, y])
                output_value = sess.run([output_layer], feed_dict={
                        self.input_layer : [datum],
                    })
                output_values.append(abs(output_value[0][0]))
            index = np.argmin(output_values)
            ys_res.append(ys[index])
        return xs, ys_res

    def get_plot_buf_color(self, sess, regionA, regionB):
        """
        Create a pyplot plot and save to buffer.
        """
        plt.figure()
        Xs, Ys, Zs = self.make_color_plot(sess, self.input_layer, self.output_layer_y)
        plt.scatter(*regionA, marker='x', label='Region A')
        plt.scatter(*regionB, marker='+', label='Region B')
        plt.pcolor(Xs, Ys, Zs, alpha=0.2, cmap='winter')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    # test
    def get_plot_buf(self, sess, regionA, regionB):
        """
        Create a pyplot plot and save to buffer.
        """
        plt.figure()
        xs, ys = self.make_decision_line(sess, self.input_layer, self.output_layer_y)
        plt.scatter(*regionA, marker='x', label='Region A')
        plt.scatter(*regionB, marker='+', label='Region B')
        plt.plot(xs, ys)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    def run(self, regionA, regionB, data_test, labels_test):
        '''
        Conduct classification experiment as the book shows
        '''
        self.min_x, self.max_x, self.min_y, self.max_y = circle.get_max_min(regionA, regionB)
        if self.normalize:
            avg, std = circle.get_avg_std(regionA, regionB)
            self.avg = avg
            self.std = std
            data_test = (np.array(data_test) - self.avg) / self.std
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('./test', sess.graph)
            for epoch_i in range(self.max_epoch):

                plot_buf = self.get_plot_buf_color(sess, regionA, regionB)

                '''
                test validation
                '''
                w_summary = sess.run(
                        self.summary,
                    feed_dict={
                        self.input_layer : data_test,
                        self.label_layer : labels_test,
                        self.plot_buf_ph : plot_buf.getvalue(),
                    })
                writer.add_summary(w_summary, epoch_i)

                '''
                online training
                '''
                epoch = circle.make_a_epoch(regionA, regionB)
                for iter_i_in_epoch, (datum, label) in enumerate(epoch):
                    if self.normalize:
                        datum = (np.array(datum) - self.avg) / self.std
                    self.loss_value, _ = sess.run([
                        self.loss,
                        self.train_step,
                    ], feed_dict={
                        self.input_layer : [datum],
                        self.label_layer : [[label]],
                    })

            writer.close()
        return

class circleClf2(circleClf):

    def set_learning_rate(self):
        self.starter_learning_rate = 1e-1
        self.end_learning_rate = 1e-3
        decay_rate = math.pow((self.end_learning_rate/self.starter_learning_rate), 1./self.max_epoch)
        self.learning_rate = tf.train.exponential_decay(
            self.starter_learning_rate,
            global_step=self.global_step,
            decay_steps=2000,
            decay_rate=decay_rate,
        )
        return

    def get_tag(self):
        return 'exponential' + ('_normalized' if self.normalize else '_not_normalized')

class circleClf3(circleClf2):

    def get_tag(self):
        return 'initialization_optimized' + ('_normalized' if self.normalize else '_not_normalized')

    def initializer(self, shape, m):
        lim = get_uniform_range(m)
        return tf.random_uniform(shape, minval=-lim, maxval=lim)

class circleClf4(circleClf2):

    def get_tag(self):
        return 'relu' + ('_normalized' if self.normalize else '_not_normalized')

    def act_fn(self, in_tensor):
        return tf.nn.relu(in_tensor)

class circleClf5(circleClf4, circleClf3):

    def get_tag(self):
        return 'relu_initialization_optimized' + ('_normalized' if self.normalize else '_not_normalized')
