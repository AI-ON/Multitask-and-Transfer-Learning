import os.path
import time
import numpy as np
import random

from chainer import (Variable, Chain, serializers, optimizers, report)

import chainer.functions as F
import chainer.links as L
import chainer.initializers as I

from components.conv_gru import ConvGRU2D
from components.stateless_conv_gru import StatelessConvGRU2D
from components.embedding_conv2d import EmbeddingConv2D

class PredictiveAutoencoder(Chain):
    def __init__(self, action_space=18):
        w = I.HeNormal()
        super(PredictiveAutoencoder, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                in_channels=3,
                out_channels=16,
                ksize=8,
                stride=4,
                initialW=w,
            )
            self.embed_conv2d = EmbeddingConv2D(
                embed_size=action_space,
                in_channels=3,
                out_channels=16,
                ksize=8,
                stride=4,
                initialW=w,
            )
            self.conv2 = L.Convolution2D(
                in_channels=32,
                out_channels=32,
                ksize=4,
                stride=2,
                initialW=w,
            )
            self.conv3 = L.Convolution2D(
                32,
                out_channels=32,
                ksize=1,
                pad=0,
                initialW=w,
            )
            self.conv_gru1 = ConvGRU2D(
                32,
                out_channels=64,
                ksize=1,
                init=w,
                inner_init=w,
            )
            self.linear1 = L.Linear(
                None,
                256,
                initialW=w,
            )
            self.linear2 = L.Linear(
                256,
                out_size=action_space,
                initialW=w,
            )
            self.deconv1 = L.Deconvolution2D(
                64,
                out_channels=32,
                ksize=4,
                stride=2,
                outsize=(39, 51),
                initialW=w,
            )
            self.deconv2 = L.Deconvolution2D(
                32,
                out_channels=3,
                ksize=8,
                stride=4,
                outsize=(160, 210),
                initialW=w,
            )


    def __call__(self, x, action):
        h1 = F.relu(self.conv1(x))
        index = F.expand_dims(np.array(action, dtype=np.int32), axis=0)
        h2 = F.relu(self.embed_conv2d(index, x))
        h = F.concat((h1, h2), axis=1)  # Glue together the action convolutions
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv_gru1(h))

        h_img = F.relu(self.deconv1(h))
        h_img = self.deconv2(h_img)

        h_action = F.relu(self.linear1(h))
        h_action = self.linear2(h_action)

        return h_img, h_action


class Classifier(Chain):
    def __init__(self, predictor, weight=0.5):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.predictor = predictor
            self.weight = float(weight)
            self.y_image = None
            self.y_action = None

    def __call__(self, x_image, t_image, x_action, t_action):
        self.y_image, self.y_action = self.predictor(x_image, x_action)
        print("Predicted action: {}, it was actually {}".format(
            F.argmax(self.y_action, axis=1).data, t_action))
        image_loss = F.mean_squared_error(self.y_image, t_image)
        action_loss = F.softmax_cross_entropy(
            self.y_action,
            F.expand_dims(np.array(t_action, dtype=np.int32), axis=0),
        )
        print('Image loss', image_loss, ', Action loss:', action_loss)
        return self.weight * image_loss + (1.0 - self.weight) * action_loss


class PredictorAgent(object):
    def __init__(self, save_dir,
                 name=None,
                 load_saved=True,
                 classifier_weight=0.5,
                 backprop_rounds=10):
        self.name = name or 'predictive_autoencoder'
        self.save_dir = save_dir
        self.backprop_rounds = backprop_rounds

        self.model = PredictiveAutoencoder(
        )

        self.classifier = Classifier(self.model, weight=classifier_weight)

        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

        # Trainer state
        self.last_action = 0
        self.last_obs = None
        self.i = 0
        self._predicted_image = None
        self.predicted_action = None
        self.loss = 0

        if load_saved and os.path.exists(self._model_filename()):
            self.load(self.save_dir)

    def initialize_state(self, first_obs):
        self.last_obs = first_obs

    def _model_filename(self):
        return self.save_dir + self.name + '.model'

    def _opti_filename(self):
        return self.save_dir + self.name + '.optimizer'

    def save(self):
        # TODO: use hd5 groups to put everything in one file
        # TODO: serialize the classifier, the weight is being lost
        serializers.save_hdf5(self._model_filename(), self.model)
        serializers.save_hdf5(self._opti_filename(), self.optimizer)
        return time.time()

    def load(self, save_dir):
        serializers.load_hdf5(self._model_filename(), self.model)
        serializers.load_hdf5(self._opti_filename(), self.optimizer)

    @property
    def predicted_image(self):
        # The transpose is because OpenAI gym and chainer have
        # different depth conventions
        return F.transpose(self._predicted_image[0])

    def describe_model(self):
        # TODO: fix this method
        graph = cg.build_computational_graph(model)

        dot_name = DIAGRAM_DIR + model_name + '.dot'
        png_name = DIAGRAM_DIR + model_name + '.png'

        with open(dot_name, 'w') as o:
            o.write(graph.dump())

            subprocess.call(['dot', '-Tpng', dot_name, '-o', png_name])


    def __call__(self, obs, reward):
        '''Takes in observation (processed) and returns the next action to
        execute'''
        self.i += 1
        # policy!
        action = self.i % 6

        self.loss += self.classifier(self.last_obs, obs, self.last_action, action)
        self._predicted_image = self.classifier.y_image
        self.predicted_action = self.classifier.y_action

        self.last_action = action
        self.last_obs = obs

        if self.i % self.backprop_rounds == 0:
            self.model.cleargrads()
            self.loss.backward()
            self.loss.unchain_backward()
            self.optimizer.update()
            report({'loss': self.loss})
        return action

def to_one_hot(size, index):
    '''Converts an int into a one-hot array'''
    arr = np.zeros(size, dtype=np.int32)
    arr[index] = 1
    return arr
