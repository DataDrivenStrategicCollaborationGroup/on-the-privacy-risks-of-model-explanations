# Most code in this file is taken from the code for the paper "Understanding Black-box Predictions
# via Influence Functions" for mPang Wei Koh and Percy Liang the original code can be found at
#  http://bit.ly/gt-influence and http://bit.ly/cl-influence. The code was adapted to Python 3 and to follow the PEP 8
# format. It can be used to transform the fishdog dataset. While the functions in this file can do a lot more nothing
# else has been tested and we recommend using the original code for further experiments.
import tensorflow as tf
from tensorflow import gradients
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from sklearn import linear_model
import os
import numpy as np
import math
from tensorflow.contrib.learn.python.learn.datasets import base
from keras.models import Model
from keras import layers
from keras.layers import Activation, Flatten, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D,\
    GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend
# noinspection PyProtectedMember
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing import image
from time import time
import warnings
import pickle
from constants import DATA_PATH


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/'\
                'download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/' \
                      'releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


BASE_DIR = DATA_PATH + 'fishdog'


class GenericNeuralNet(object):
    """
    Multi-class classification.
    """

    def __init__(self, **kwargs):
        np.random.seed(0)
        tf.set_random_seed(0)

        self.batch_size = kwargs.pop('batch_size')
        self.data_sets = kwargs.pop('data_sets')
        self.train_dir = kwargs.pop('train_dir', 'output')
        self.model_name = kwargs.pop('model_name')
        self.num_classes = kwargs.pop('num_classes')
        self.initial_learning_rate = kwargs.pop('initial_learning_rate')
        self.decay_epochs = kwargs.pop('decay_epochs')

        if 'keep_probs' in kwargs:
            self.keep_probs = kwargs.pop('keep_probs')
        else:
            self.keep_probs = None

        if 'mini_batch' in kwargs:
            self.mini_batch = kwargs.pop('mini_batch')
        else:
            self.mini_batch = True

        if 'damping' in kwargs:
            self.damping = kwargs.pop('damping')
        else:
            self.damping = 0.0

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        # Initialize session
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)
        backend.set_session(self.sess)

        # Setup input
        self.input_placeholder, self.labels_placeholder = self.placeholder_inputs()
        self.num_train_examples = self.data_sets.train.labels.shape[0]
        self.num_test_examples = self.data_sets.test.labels.shape[0]

        # Setup inference and training
        if self.keep_probs is not None:
            self.keep_probs_placeholder = tf.placeholder(tf.float32, shape=2)
            self.logits = self.inference(self.input_placeholder, probs=self.keep_probs_placeholder)
        elif hasattr(self, 'inference_needs_labels'):
            self.logits = self.inference(self.input_placeholder, labels=self.labels_placeholder)
        else:
            self.logits = self.inference(self.input_placeholder)

        self.total_loss, self.loss_no_reg, self.indiv_loss_no_reg = self.loss(
            self.logits,
            self.labels_placeholder)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.Variable(self.initial_learning_rate, name='learning_rate', trainable=False)
        self.learning_rate_placeholder = tf.placeholder(tf.float32)
        self.update_learning_rate_op = tf.assign(self.learning_rate, self.learning_rate_placeholder)

        self.train_op = self.get_train_op(self.total_loss, self.global_step, self.learning_rate)
        self.train_sgd_op = self.get_train_sgd_op(self.total_loss, self.global_step, self.learning_rate)
        self.accuracy_op = self.get_accuracy_op(self.logits, self.labels_placeholder)
        self.preds = self.predictions(self.logits)

        # Setup misc
        self.saver = tf.train.Saver()

        # Setup gradients and Hessians
        self.params = self.get_all_params()
        self.grad_total_loss_op = tf.gradients(self.total_loss, self.params)
        self.grad_loss_no_reg_op = tf.gradients(self.loss_no_reg, self.params)
        self.v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]
        self.u_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params]

        self.hessian_vector = hessian_vector_product(self.total_loss, self.params, self.v_placeholder)

        self.grad_loss_wrt_input_op = tf.gradients(self.total_loss, self.input_placeholder)

        # Because tf.gradients auto accumulates, we probably don't need the add_n (or even reduce_sum)
        # noinspection PyUnresolvedReferences
        self.influence_op = tf.add_n(
            [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in
             zip(self.grad_total_loss_op, self.v_placeholder)])

        self.grad_influence_wrt_input_op = tf.gradients(self.influence_op, self.input_placeholder)

        self.checkpoint_file = os.path.join(self.train_dir, "%s-checkpoint" % self.model_name)

        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.vec_to_list = self.get_vec_to_list_fn()
        self.adversarial_loss, self.indiv_adversarial_loss = self.adversarial_loss(self.logits, self.labels_placeholder)
        if self.adversarial_loss is not None:
            self.grad_adversarial_loss_op = tf.gradients(self.adversarial_loss, self.params)
        self.num_params = None

    def placeholder_inputs(self):
        raise NotImplementedError

    def inference(self, inputs, labels=None, probs=None):
        raise NotImplementedError

    @staticmethod
    def get_all_params():
        raise NotImplementedError

    @staticmethod
    def predictions(logits):
        raise NotImplementedError

    def get_vec_to_list_fn(self):
        params_val = self.sess.run(self.params)
        self.num_params = len(np.concatenate(params_val))
        print('Total number of parameters: %s' % self.num_params)

        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos: cur_pos + len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list

    def reset_datasets(self):
        for data_set in self.data_sets:
            if data_set is not None:
                data_set.reset_batch()

    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels
        }
        return feed_dict

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx]
        }
        return feed_dict

    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def fill_feed_dict_manual(self, x, y):
        x = np.array(x)
        y = np.array(y)
        input_feed = x.reshape(len(y), -1)
        labels_feed = y.reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def minibatch_mean_eval(self, ops, data_set):

        num_examples = data_set.num_examples
        assert num_examples % self.batch_size == 0
        num_iter = int(num_examples / self.batch_size)

        self.reset_datasets()

        ret = []
        for i in range(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(data_set)
            ret_temp = self.sess.run(ops, feed_dict=feed_dict)

            if len(ret) == 0:
                for b in ret_temp:
                    if isinstance(b, list):
                        ret.append([c / float(num_iter) for c in b])
                    else:
                        ret.append([b / float(num_iter)])
            else:
                for counter, b in enumerate(ret_temp):
                    if isinstance(b, list):
                        ret[counter] = [a + (c / float(num_iter)) for (a, c) in zip(ret[counter], b)]
                    else:
                        ret[counter] += (b / float(num_iter))

        return ret

    def print_model_eval(self):
        params_val = self.sess.run(self.params)

        if self.mini_batch:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.minibatch_mean_eval(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op],
                self.data_sets.train)

            test_loss_val, test_acc_val = self.minibatch_mean_eval(
                [self.loss_no_reg, self.accuracy_op],
                self.data_sets.test)

        else:
            grad_loss_val, loss_no_reg_val, loss_val, train_acc_val = self.sess.run(
                [self.grad_total_loss_op, self.loss_no_reg, self.total_loss, self.accuracy_op],
                feed_dict=self.all_train_feed_dict)

            test_loss_val, test_acc_val = self.sess.run(
                [self.loss_no_reg, self.accuracy_op],
                feed_dict=self.all_test_feed_dict)

        print('Train loss (w reg) on all data: %s' % loss_val)
        print('Train loss (w/o reg) on all data: %s' % loss_no_reg_val)

        print('Test loss (w/o reg) on all data: %s' % test_loss_val)
        print('Train acc on all data:  %s' % train_acc_val)
        print('Test acc on all data:   %s' % test_acc_val)

        print('Norm of the mean of gradients: %s' % np.linalg.norm(np.concatenate(grad_loss_val)))
        print('Norm of the params: %s' % np.linalg.norm(np.concatenate(params_val)))

    def retrain(self, num_steps, feed_dict):
        for step in range(num_steps):
            self.sess.run(self.train_op, feed_dict=feed_dict)

    def update_learning_rate(self, step):
        assert self.num_train_examples % self.batch_size == 0
        num_steps_in_epoch = self.num_train_examples / self.batch_size
        epoch = step // num_steps_in_epoch

        if epoch < self.decay_epochs[0]:
            multiplier = 1
        elif epoch < self.decay_epochs[1]:
            multiplier = 0.1
        else:
            multiplier = 0.01

        self.sess.run(
            self.update_learning_rate_op,
            feed_dict={self.learning_rate_placeholder: multiplier * self.initial_learning_rate})

    def train(self, num_steps,
              iter_to_switch_to_batch=20000,
              iter_to_switch_to_sgd=40000,
              save_checkpoints=True, verbose=True):
        """
        Trains a model for a specified number of steps.
        """
        if verbose:
            print('Training for %s steps' % num_steps)

        sess = self.sess

        for step in range(num_steps):
            self.update_learning_rate(step)

            start_time = time()

            if step < iter_to_switch_to_batch:
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train)
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

            elif step < iter_to_switch_to_sgd:
                feed_dict = self.all_train_feed_dict
                _, loss_val = sess.run([self.train_op, self.total_loss], feed_dict=feed_dict)

            else:
                feed_dict = self.all_train_feed_dict
                _, loss_val = sess.run([self.train_sgd_op, self.total_loss], feed_dict=feed_dict)

            duration = time() - start_time

            if verbose:
                if step % 1000 == 0:
                    # Print status to stdout.
                    print('Step %d: loss = %.8f (%.3f sec)' % (step, loss_val, duration))

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 100000 == 0 or (step + 1) == num_steps:
                if save_checkpoints:
                    self.saver.save(sess, self.checkpoint_file, global_step=step)
                if verbose:
                    self.print_model_eval()

    def load_checkpoint(self, iter_to_load, do_checks=True):
        checkpoint_to_load = "%s-%s" % (self.checkpoint_file, iter_to_load)
        self.saver.restore(self.sess, checkpoint_to_load)

        if do_checks:
            print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)
            self.print_model_eval()

    @staticmethod
    def get_train_op(total_loss, global_step, learning_rate):
        """
        Return train_op
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op

    @staticmethod
    def get_train_sgd_op(total_loss, global_step, learning_rate=tf.Variable(0.001)):
        """
        Return train_sgd_op
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        return train_op

    @staticmethod
    def get_accuracy_op(logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        correct = tf.nn.in_top_k(logits, labels, 1)
        # noinspection PyUnresolvedReferences
        return tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels)[0]

    def loss(self, logits, labels):

        labels = tf.one_hot(labels, depth=self.num_classes)
        # correct_prob = tf.reduce_sum(tf.multiply(labels, tf.nn.softmax(logits)), reduction_indices=1)
        # noinspection PyUnresolvedReferences
        cross_entropy = - tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(logits)), reduction_indices=1)

        indiv_loss_no_reg = cross_entropy
        # noinspection PyUnresolvedReferences
        loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg

    def adversarial_loss(self, logits, labels):
        # Computes sum of log(1 - p(y = true|x))
        # No regularization (because this is meant to be computed on the test data)

        labels = tf.one_hot(labels, depth=self.num_classes)
        wrong_labels = (labels - 1) * -1  # Flips 0s and 1s
        wrong_labels_bool = tf.reshape(tf.cast(wrong_labels, tf.bool), [-1, self.num_classes])

        wrong_logits = tf.reshape(tf.boolean_mask(logits, wrong_labels_bool), [-1, self.num_classes - 1])
        # noinspection PyUnresolvedReferences
        indiv_adversarial_loss = tf.reduce_logsumexp(
            wrong_logits, reduction_indices=1) - tf.reduce_logsumexp(logits, reduction_indices=1)
        # noinspection PyUnresolvedReferences
        adversarial_loss = tf.reduce_mean(indiv_adversarial_loss)

        return adversarial_loss, indiv_adversarial_loss  # , indiv_wrong_prob

    def update_feed_dict_with_v_placeholder(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder, vec):
            feed_dict[pl_block] = vec_block
        return feed_dict

    def get_inverse_hvp(self, v, approx_type='cg', approx_params=None, verbose=True):
        assert approx_type in ['cg', 'lissa']
        if approx_type == 'lissa':
            return self.get_inverse_hvp_lissa(v, **approx_params)
        elif approx_type == 'cg':
            return self.get_inverse_hvp_cg(v, verbose)

    def get_inverse_hvp_cg(self, v, verbose):
        raise NotImplementedError

    def get_inverse_hvp_lissa(self, v,
                              batch_size=None,
                              scale=10, damping=0.0, num_samples=1, recursion_depth=10000):
        """
        This uses mini-batching; uncomment code for the single sample case.
        """
        inverse_hvp = None
        print_iter = recursion_depth / 10

        for i in range(num_samples):
            # samples = np.random.choice(self.num_train_examples, size=recursion_depth)

            cur_estimate = v

            for j in range(recursion_depth):

                # feed_dict = fill_feed_dict_with_one_ex(
                #   data_set,
                #   images_placeholder,
                #   labels_placeholder,
                #   samples[j])
                feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)

                feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, cur_estimate)
                hessian_vector_val = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
                cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in
                                zip(v, cur_estimate, hessian_vector_val)]

                # Update: v + (I - Hessian_at_x) * cur_estimate
                if (j % print_iter == 0) or (j == recursion_depth - 1):
                    print("Recursion at depth %s: norm is %.8lf" % (j, np.linalg.norm(np.concatenate(cur_estimate))))

            if inverse_hvp is None:
                inverse_hvp = [b / scale for b in cur_estimate]
            else:
                inverse_hvp = [a + b / scale for (a, b) in zip(inverse_hvp, cur_estimate)]

        inverse_hvp = [a / num_samples for a in inverse_hvp]
        return inverse_hvp

    def minibatch_hessian_vector_val(self, v):

        num_examples = self.num_train_examples
        if self.mini_batch:
            batch_size = 100
            assert num_examples % batch_size == 0
        else:
            batch_size = self.num_train_examples

        num_iter = int(num_examples / batch_size)

        self.reset_datasets()
        hessian_vector_val = None
        for i in range(num_iter):
            feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=batch_size)
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a, b) in
                                      zip(hessian_vector_val, hessian_vector_val_temp)]

        hessian_vector_val = [a + self.damping * b for (a, b) in zip(hessian_vector_val, v)]

        return hessian_vector_val

    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)

        return get_fmin_loss

    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return np.concatenate(hessian_vector_val) - np.concatenate(v)

        return get_fmin_grad

    def get_fmin_hvp(self, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(p))

        return np.concatenate(hessian_vector_val)

    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)

        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            local_v = self.vec_to_list(x)
            idx_to_remove = 5

            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(np.concatenate(local_v),
                                         np.concatenate(train_grad_loss_val)) / self.num_train_examples

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback

    def get_test_grad_loss_no_reg_val(self, test_indices, batch_size=100, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        else:
            raise ValueError('Loss must be specified')

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i + 1) * batch_size, len(test_indices)))
                test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end - start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end - start) for (a, b) in
                                                 zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a / len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]

        return test_grad_loss_no_reg_val

    def get_example_set_grad_loss_no_reg_val(self, example_set, batch_size=100, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_loss_no_reg_op
        elif loss_type == 'adversarial_loss':
            op = self.grad_adversarial_loss_op
        else:
            raise ValueError('Loss must be specified')

        if example_set is not None:
            num_iter = int(np.ceil(len(example_set.x) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i + 1) * batch_size, len(example_set.x)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(example_set, range(start, end))

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end - start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end - start) for (a, b) in
                                                 zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a / len(example_set.x) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]

        return test_grad_loss_no_reg_val

    def get_influence_on_test_loss(self, test_indices, train_idx,
                                   approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
                                   loss_type='normal_loss',
                                   x=None, y=None, verbose=False, save=False):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order

        if train_idx is None:
            if (x is None) or (y is None):
                raise ValueError('X and Y must be specified if using phantom points.')
            if x.shape[0] != len(y):
                raise ValueError('X and Y must have the same length.')
        else:
            if (x is not None) or (y is not None):
                raise ValueError('X and Y cannot be specified if train_idx is specified.')

        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        if verbose:
            print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))
        if not force_refresh and os.path.exists(approx_filename):
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose:
                print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params, verbose=verbose)
            if save:
                np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose:
                print('Saved inverse HVP to %s' % approx_filename)

        duration = time() - start_time
        if verbose:
            print('Inverse HVP took %s sec' % duration)

        start_time = time()
        if train_idx is None:
            num_to_remove = len(y)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(x[counter, :], [y[counter]])
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        else:
            num_to_remove = len(train_idx)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(train_idx):
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        duration = time() - start_time
        if verbose:
            print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs

    def get_influence_on_example_set_loss(self, example_set, train_idx,
                                          approx_type='cg', approx_params=None, force_refresh=True,
                                          test_description=None,
                                          loss_type='normal_loss',
                                          x=None, y=None, verbose=False):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order
        if test_description is None:
            test_description = ["Example"]

        if train_idx is None:
            if (x is None) or (y is None):
                raise ValueError('X and Y must be specified if using phantom points.')
            if x.shape[0] != len(y):
                raise ValueError('X and Y must have the same length.')
        else:
            if (x is not None) or (y is not None):
                raise ValueError('X and Y cannot be specified if train_idx is specified.')

        test_grad_loss_no_reg_val = self.get_example_set_grad_loss_no_reg_val(example_set, loss_type=loss_type)

        if verbose:
            print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time()

        if test_description is None:
            test_description = ["Example"]

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and not force_refresh:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose:
                print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params, verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose:
                print('Saved inverse HVP to %s' % approx_filename)

        duration = time() - start_time
        if verbose:
            print('Inverse HVP took %s sec' % duration)

        start_time = time()
        if train_idx is None:
            num_to_remove = len(y)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(x[counter, :], [y[counter]])
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        else:
            num_to_remove = len(train_idx)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(train_idx):
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        duration = time() - start_time
        if verbose:
            print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))

        return predicted_loss_diffs

    def get_grad_of_influence_wrt_input(self, train_indices, test_indices,
                                        approx_type='cg', approx_params=None, force_refresh=True, verbose=True,
                                        test_description=None,
                                        loss_type='normal_loss'):
        """
        If the loss goes up when you remove a point, then it was a helpful point.
        So positive influence = helpful.
        If we move in the direction of the gradient, we make the influence even more positive,
        so even more helpful.
        Thus if we want to make the test point more wrong, we have to move in the opposite direction.
        """

        # Calculate v_placeholder (gradient of loss at test point)
        test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)

        if verbose:
            print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))

        if os.path.exists(approx_filename) and not force_refresh:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            if verbose:
                print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params,
                verbose=verbose)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            if verbose:
                print('Saved inverse HVP to %s' % approx_filename)

        duration = time() - start_time
        if verbose:
            print('Inverse HVP took %s sec' % duration)

        grad_influence_wrt_input_val = None

        for counter, train_idx in enumerate(train_indices):
            # Put in the train example in the feed dict
            grad_influence_feed_dict = self.fill_feed_dict_with_one_ex(
                self.data_sets.train,
                train_idx)

            self.update_feed_dict_with_v_placeholder(grad_influence_feed_dict, inverse_hvp)

            # Run the grad op with the feed dict
            current_grad_influence_wrt_input_val = \
                self.sess.run(self.grad_influence_wrt_input_op, feed_dict=grad_influence_feed_dict)[0][0, :]

            if grad_influence_wrt_input_val is None:
                grad_influence_wrt_input_val = np.zeros([len(train_indices), len(current_grad_influence_wrt_input_val)])

            grad_influence_wrt_input_val[counter, :] = current_grad_influence_wrt_input_val

        return grad_influence_wrt_input_val

    def update_train_x(self, new_train_x):
        assert np.all(new_train_x.shape == self.data_sets.train.x.shape)
        new_train = DataSet(new_train_x, np.copy(self.data_sets.train.labels))
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.reset_datasets()

    def update_train_x_y(self, new_train_x, new_train_y):
        new_train = DataSet(new_train_x, new_train_y)
        self.data_sets = base.Datasets(train=new_train, validation=self.data_sets.validation, test=self.data_sets.test)
        self.all_train_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.train)
        self.num_train_examples = len(new_train_y)
        self.reset_datasets()

    def update_test_x_y(self, new_test_x, new_test_y):
        new_test = DataSet(new_test_x, new_test_y)
        self.data_sets = base.Datasets(train=self.data_sets.train, validation=self.data_sets.validation, test=new_test)
        self.all_test_feed_dict = self.fill_feed_dict_with_all_ex(self.data_sets.test)
        self.num_test_examples = len(new_test_y)
        self.reset_datasets()


class BinaryInceptionModel(GenericNeuralNet):

    def get_inverse_hvp_cg(self, v, verbose):
        pass

    def __init__(self, img_side, num_channels, weight_decay, **kwargs):
        self.weight_decay = weight_decay

        self.img_side = img_side
        self.num_channels = num_channels
        self.input_dim = img_side * img_side * num_channels
        self.num_features = 2048  # Hardcoded for inception. For some reason Flatten() doesn't register num_features.
        self.inception_model = None
        self.inception_features = None
        self.weights = None
        self.W_placeholder = None
        super(BinaryInceptionModel, self).__init__(**kwargs)

        self.load_inception_weights()
        # Do we need to set trainable to False?
        # We might be unnecessarily blowing up the graph by including all of the train operations
        # needed for the inception network.

        self.set_params_op = self.set_params()

        c = 1.0 / (self.num_train_examples * self.weight_decay)
        self.sklearn_model = linear_model.LogisticRegression(
            C=c,
            tol=1e-8,
            fit_intercept=False,
            solver='lbfgs',
            # multi_class='multinomial',
            warm_start=True,
            max_iter=1000)

        c_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=c_minus_one,
            tol=1e-8,
            fit_intercept=False,
            solver='lbfgs',
            # multi_class='multinomial',
            warm_start=True,
            max_iter=1000)

    @staticmethod
    def get_all_params():
        all_params = []
        for layer in ['softmax_linear']:
            # for var_name in ['weights', 'biases']:
            for var_name in ['weights']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                all_params.append(temp_tensor)
        return all_params

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=None,
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def fill_feed_dict_with_all_ex(self, data_set):
        feed_dict = {
            self.input_placeholder: data_set.x,
            self.labels_placeholder: data_set.labels,
            backend.learning_phase(): 0
        }
        return feed_dict

    def fill_feed_dict_with_all_but_one_ex(self, data_set, idx_to_remove):
        num_examples = data_set.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        feed_dict = {
            self.input_placeholder: data_set.x[idx, :],
            self.labels_placeholder: data_set.labels[idx],
            backend.learning_phase(): 0
        }
        return feed_dict

    def fill_feed_dict_with_batch(self, data_set, batch_size=0):
        if batch_size is None:
            return self.fill_feed_dict_with_all_ex(data_set)
        elif batch_size == 0:
            batch_size = self.batch_size

        input_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            backend.learning_phase(): 0
        }
        return feed_dict

    def fill_feed_dict_with_some_ex(self, data_set, target_indices):
        input_feed = data_set.x[target_indices, :].reshape(len(target_indices), -1)
        labels_feed = data_set.labels[target_indices].reshape(-1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            backend.learning_phase(): 0
        }
        return feed_dict

    def fill_feed_dict_with_one_ex(self, data_set, target_idx):
        input_feed = data_set.x[target_idx, :].reshape(1, -1)
        labels_feed = data_set.labels[target_idx].reshape(1)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
            backend.learning_phase(): 0
        }
        return feed_dict

    def load_inception_weights(self):
        # Replace this with a local copy for reproducibility
        # TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/
        # download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        # weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                         TF_WEIGHTS_PATH_NO_TOP,
        #                         cache_subdir='models',
        #                         md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
        weights_path = 'inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.inception_model.load_weights(weights_path)

    def inference(self, inputs, labels=None, probs=None):
        reshaped_input = tf.reshape(inputs, [-1, self.img_side, self.img_side, self.num_channels])
        self.inception_model = inception_v3(include_top=False, weights='imagenet',
                                            input_tensor=reshaped_input, input_shape=(299, 299, 3))

        raw_inception_features = self.inception_model.output
        print(raw_inception_features.shape)
        pooled_inception_features = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(raw_inception_features)
        # pooled_inception_features = AveragePooling2D((6, 6), strides=(6, 6), name='avg_pool')(raw_inception_features)
        self.inception_features = Flatten(name='flatten')(pooled_inception_features)

        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights',
                [self.num_features],
                stddev=1.0 / math.sqrt(float(self.num_features)),
                wd=self.weight_decay)

            logits = tf.matmul(self.inception_features, tf.reshape(weights, [-1, 1]))
            zeros = tf.zeros_like(logits)
            logits_with_zeros = tf.concat([zeros, logits], 1)

        self.weights = weights

        return logits_with_zeros

    @staticmethod
    def predictions(logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds

    def set_params(self):
        # See if we can automatically infer weight shape
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=[self.num_features],
            name='W_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)
        return [set_weights]

    def retrain(self, num_steps, feed_dict):
        self.train_with_lbfgs(
            feed_dict=feed_dict,
            save_checkpoints=False,
            verbose=False)

    def train(self, num_steps=None,
              iter_to_switch_to_batch=None,
              iter_to_switch_to_sgd=None,
              save_checkpoints=True, verbose=True):

        self.train_with_lbfgs(
            feed_dict=self.all_train_feed_dict,
            save_checkpoints=save_checkpoints,
            verbose=verbose)

    def train_with_sgd(self, **kwargs):
        super(BinaryInceptionModel, self).train(**kwargs)

    def minibatch_inception_features(self, feed_dict):

        num_examples = feed_dict[self.input_placeholder].shape[0]
        batch_size = 100
        num_iter = int(np.ceil(num_examples / batch_size))

        ret = np.zeros([num_examples, self.num_features])

        batch_feed_dict = {backend.learning_phase(): 0}

        for i in range(num_iter):
            start = i * batch_size
            end = (i + 1) * batch_size
            if end > num_examples:
                end = num_examples

            batch_feed_dict[self.input_placeholder] = feed_dict[self.input_placeholder][start:end]
            batch_feed_dict[self.labels_placeholder] = feed_dict[self.labels_placeholder][start:end]
            ret[start:end, :] = self.sess.run(self.inception_features, feed_dict=batch_feed_dict)

        return ret

    def train_with_lbfgs(self, feed_dict, save_checkpoints=True, verbose=True):
        # More sanity checks to see if predictions are the same?

        # x_train = feed_dict[self.input_placeholder]
        # x_train = self.sess.run(self.inception_features, feed_dict=feed_dict)
        x_train = self.minibatch_inception_features(feed_dict)

        y_train = feed_dict[self.labels_placeholder]
        num_train_examples = len(y_train)
        assert len(y_train.shape) == 1
        assert x_train.shape[0] == y_train.shape[0]

        if num_train_examples == self.num_train_examples:
            print('Using normal model')
            model = self.sklearn_model
        elif num_train_examples == self.num_train_examples - 1:
            print('Using model minus one')
            model = self.sklearn_model_minus_one
        else:
            raise ValueError("feed_dict has incorrect number of training examples")

        model.fit(x_train, y_train)
        # sklearn returns coefficients in shape num_classes x num_features
        # whereas our weights are defined as num_features x num_classes
        # so we have to transpose them first.
        w = np.reshape(model.coef_.T, -1)
        # b = model.intercept_

        params_feed_dict = {self.W_placeholder: w}

        # params_feed_dict[self.b_placeholder] = b
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if save_checkpoints:
            self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            print('LBFGS training took %s iter.' % model.n_iter_)
            print('After training with LBFGS: ')
            self.print_model_eval()

    def load_weights_from_disk(self, weights_filename, do_check=True, do_save=True):
        w = np.load('%s' % weights_filename)

        params_feed_dict = {self.W_placeholder: w}
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if do_save:
            self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        print('Loaded weights from disk.')
        if do_check:
            self.print_model_eval()


class DataSet(object):

    def __init__(self, x, labels):

        if len(x.shape) > 2:
            x = np.reshape(x, [x.shape[0], -1])

        assert(x.shape[0] == labels.shape[0])

        x = x.astype(np.float32)

        self._x = x
        self._x_batch = np.copy(x)
        self._labels = labels
        self._labels_batch = np.copy(labels)
        self._num_examples = x.shape[0]
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def reset_batch(self):
        self._index_in_epoch = 0
        self._x_batch = np.copy(self._x)
        self._labels_batch = np.copy(self._labels)

    def next_batch(self, batch_size):
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:

            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._x_batch = self._x_batch[perm, :]
            self._labels_batch = self._labels_batch[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self._x_batch[start:end], self._labels_batch[start:end]


def fill(y, idx, label, img_path, img_side):
    img = image.load_img(img_path, target_size=(img_side, img_side))
    x = image.img_to_array(img)
    x[idx, ...] = x
    y[idx] = label


def load_animals(num_train_ex_per_class=300,
                 num_test_ex_per_class=100,
                 num_valid_ex_per_class=0,
                 classes=None,
                 ):

    num_channels = 3
    img_side = 299

    if num_valid_ex_per_class == 0:
        valid_str = ''
    else:
        valid_str = ''

    if classes is None:
        classes = ['dog', 'cat', 'bird', 'fish', 'horse', 'monkey', 'zebra', 'panda', 'lemur', 'wombat']
        data_filename = os.path.join(BASE_DIR, 'dataset_train-%s_test-%s%s.npz' % (num_train_ex_per_class,
                                                                                   num_test_ex_per_class, valid_str))
    else:
        data_filename = os.path.join(BASE_DIR, 'dataset_%s_train-%s_test-%s%s.npz' % ('-'.join(classes),
                                                                                      num_train_ex_per_class,
                                                                                      num_test_ex_per_class, valid_str))

    num_classes = len(classes)
    num_train_examples = num_train_ex_per_class * num_classes
    num_test_examples = num_test_ex_per_class * num_classes
    num_valid_examples = num_valid_ex_per_class * num_classes
    print(data_filename)
    if os.path.exists(data_filename):
        print('Loading animals from disk...')
        f = np.load(data_filename)
        x_train = f['X_train']
        x_test = f['X_test']
        y_train = f['Y_train']
        y_test = f['Y_test']

        if 'X_valid' in f:
            x_valid = f['X_valid']
        else:
            x_valid = None

        if 'Y_valid' in f:
            y_valid = f['Y_valid']
        else:
            y_valid = None

    else:
        print('Reading animals from raw images...')
        x_train = np.zeros([num_train_examples, img_side, img_side, num_channels])
        x_test = np.zeros([num_test_examples, img_side, img_side, num_channels])
        x_valid = np.zeros([num_valid_examples, img_side, img_side, num_channels])

        y_train = np.zeros([num_train_examples])
        y_test = np.zeros([num_test_examples])
        y_valid = np.zeros([num_valid_examples])

        for class_idx, class_string in enumerate(classes):
            print('class: %s' % class_string)
            # For some reason, a lot of numbers are skipped.
            i = 0
            num_filled = 0
            while num_filled < num_train_ex_per_class:
                img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
                print(img_path)
                if os.path.exists(img_path):
                    fill(y_train, num_filled + (num_train_ex_per_class * class_idx), class_idx, img_path, img_side)
                    num_filled += 1
                    print(num_filled)
                i += 1

            num_filled = 0
            while num_filled < num_test_ex_per_class:
                img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
                if os.path.exists(img_path):
                    fill(y_test, num_filled + (num_test_ex_per_class * class_idx), class_idx, img_path, img_side)
                    num_filled += 1
                    print(num_filled)
                i += 1

            num_filled = 0
            while num_filled < num_valid_ex_per_class:
                img_path = os.path.join(BASE_DIR, '%s/%s_%s.JPEG' % (class_string, class_string, i))
                if os.path.exists(img_path):
                    fill(y_valid, num_filled + (num_valid_ex_per_class * class_idx), class_idx, img_path, img_side)
                    num_filled += 1
                    print(num_filled)
                i += 1

        x_train = preprocess_input(x_train)
        x_test = preprocess_input(x_test)
        x_valid = preprocess_input(x_valid)

        np.random.seed(0)
        permutation_idx = np.arange(num_train_examples)
        np.random.shuffle(permutation_idx)
        x_train = x_train[permutation_idx, :]
        y_train = y_train[permutation_idx]
        permutation_idx = np.arange(num_test_examples)
        np.random.shuffle(permutation_idx)
        x_test = x_test[permutation_idx, :]
        y_test = y_test[permutation_idx]
        permutation_idx = np.arange(num_valid_examples)
        np.random.shuffle(permutation_idx)
        x_valid = x_valid[permutation_idx, :]
        y_valid = y_valid[permutation_idx]

        np.savez_compressed(data_filename, X_train=x_train, Y_train=y_train, X_test=x_test, Y_test=y_test,
                            X_valid=x_valid, Y_valid=y_valid)

    train = DataSet(x_train, y_train)
    if (x_valid is not None) and (y_valid is not None):
        validation = DataSet(np.asarray(x_valid), np.asarray(y_valid))
    else:
        validation = None

    test = DataSet(x_test, y_test)

    return base.Datasets(train=train, validation=validation, test=test)


"""Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

"""


def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.

    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inception_v3(include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 input_shape=None,
                 pooling=None,
                 classes=1000):
    """Instantiates the Inception v3 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    Note that the default input image size for this model is 299x299.

    Arguments:
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `channels_last` data format)
            or `(3, 299, 299)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=299,
        min_size=139,
        data_format=backend.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = Input(tensor=input_tensor, shape=input_shape)

    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='inception_v3')

    # load weights
    if weights == 'imagenet':
        if backend.image_data_format() == 'channels_first':
            if backend.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
        if include_top:
            weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
        else:
            # Replace this with a local copy for reproducibility
            # weights_path = get_file(
            #     'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
            #     WEIGHTS_PATH_NO_TOP,
            #     cache_subdir='models',
            #     md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
            weights_path = 'inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            convert_all_kernels_in_model(model)
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float32
    var = variable(
        name,
        shape,
        initializer=tf.truncated_normal_initializer(
            stddev=stddev,
            dtype=dtype))

    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def variable(name, shape, initializer):
    dtype = tf.float32
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        dtype=dtype)
    return var


def hessian_vector_product(ys, xs, v):
    """Multiply the Hessian of `ys` wrt `xs` by `v`.
    This is an efficient construction that uses a backprop-like approach
    to compute the product between the Hessian and another vector. The
    Hessian is usually too large to be explicitly computed or even
    represented, but this method allows us to at least multiply by it
    for the same big-O cost as backprop.
    Implicit Hessian-vector products are the main practical, scalable way
    of using second derivatives with neural networks. They allow us to
    do things like construct Krylov subspaces and approximate conjugate
    gradient descent.
    Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
    x, v)` will return an expression that evaluates to the same values
    as (A + A.T) `v`.
    Args:
      ys: A scalar value, or a tensor or list of tensors to be summed to
          yield a scalar.
      xs: A list of tensors that we should construct the Hessian over.
      v: A list of tensors, with the same shapes as xs, that we want to
         multiply by the Hessian.
    Returns:
      A list of tensors (or if the list would be length 1, a single tensor)
      containing the product between the Hessian and `v`.
    Raises:
      ValueError: `xs` and `v` have different length.
    """

    # Validate the input
    length = len(xs)
    if len(v) != length:
        raise ValueError("xs and v must have the same length.")

    # First backprop
    grads = gradients(ys, xs)

    # grads = xs

    assert len(grads) == length

    element_wise_product = [
        math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
        for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
    ]

    # Second backprop
    grads_with_none = gradients(element_wise_product, xs)
    return_grads = [
        grad_elem if grad_elem is not None
        else tf.zeros_like(x)
        for x, grad_elem in zip(xs, grads_with_none)]

    return return_grads


def generate_inception_features(model, poisoned_x_train_subset, labels_subset, batch_size=None):
    poisoned_train = DataSet(poisoned_x_train_subset, labels_subset)
    poisoned_data_sets = base.Datasets(train=poisoned_train, validation=None, test=None)

    if batch_size is None:
        batch_size = len(labels_subset)

    num_examples = poisoned_data_sets.train.num_examples
    assert num_examples % batch_size == 0
    num_iter = int(num_examples / batch_size)

    poisoned_data_sets.train.reset_batch()

    inception_features_val = []
    for i in range(num_iter):
        feed_dict = model.fill_feed_dict_with_batch(poisoned_data_sets.train, batch_size=batch_size)
        inception_features_val_temp = model.sess.run(model.inception_features, feed_dict=feed_dict)
        inception_features_val.append(inception_features_val_temp)

    return np.concatenate(inception_features_val)


def main():
    num_classes = 2
    num_train_ex_per_class = 900
    num_test_ex_per_class = 300

    image_data_sets = load_animals(
        num_train_ex_per_class=num_train_ex_per_class,
        num_test_ex_per_class=num_test_ex_per_class,
        classes=['dog', 'fish'])

    # Inception
    dataset_name = 'dogfish_900_300'

    # Generate inception features
    img_side = 299
    num_channels = 3
    batch_size = 100
    weight_decay = 0.001
    initial_learning_rate = 0.001
    keep_probs = None
    decay_epochs = [1000, 10000]

    tf.reset_default_graph()
    full_model_name = '%s_inception' % dataset_name
    full_model = BinaryInceptionModel(
        img_side=img_side,
        num_channels=num_channels,
        weight_decay=weight_decay,
        num_classes=num_classes,
        batch_size=batch_size,
        data_sets=image_data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=True,
        train_dir='output',
        log_dir='log',
        model_name=full_model_name)
    train_inception_features_val = generate_inception_features(
        full_model,
        image_data_sets.train.x,
        image_data_sets.train.labels,
        batch_size=batch_size)
    test_inception_features_val = generate_inception_features(
        full_model,
        image_data_sets.test.x,
        image_data_sets.test.labels,
        batch_size=batch_size)

    pickle.dump(train_inception_features_val, open("data2/fishdog/latent/latent_image_train_x.p", "bw"))
    pickle.dump(image_data_sets.train.labels, open("data2/fishdog/latent/latent/latent_image_train_y.p", "bw"))
    pickle.dump(test_inception_features_val, open("data2/fishdog/latent/latent/latent_image_test_x.p", "bw"))
    pickle.dump(image_data_sets.test.labels, open("data2/fishdog/latent/latent_image_test_y.p", "bw"))

    x = np.concatenate([train_inception_features_val, test_inception_features_val])
    y = np.concatenate([image_data_sets.train.labels, image_data_sets.test.labels])
    pickle.dump(x, open("data2/fishdog/latent/x.p", "bw"))
    pickle.dump(y, open("data2/fishdog/latent/y.p", "bw"))


if __name__ == "__main__":
    main()
