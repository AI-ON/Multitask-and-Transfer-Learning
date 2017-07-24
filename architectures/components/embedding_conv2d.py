import numpy

import chainer.links as L
import chainer.functions as F
from chainer import Variable, Link
from chainer import initializers


class EmbeddingConv2D(Link):
    '''Uses an embedding for the weights and biases of a 2d conv
    layer. As input it takes the id of the embedding to use, and the
    image to convolve.

    '''
    def __init__(self, embed_size, in_channels=None,
                 out_channels=16, ksize=(3, 3), stride=1,
                 pad=0, initialW=None):
        super(EmbeddingConv2D, self).__init__()

        self.embed_size = embed_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = _pair(ksize)
        self.stride = _pair(stride)
        self.pad = _pair(pad)

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            vec_size = self.out_channels * self.in_channels * self.kh * self.kw
            self.W_embedding = L.EmbedID(embed_size, vec_size, initialW=W_initializer)
            self.b_embedding = L.EmbedID(embed_size, out_channels)

    def __call__(self, id, x):
        W = self.W_embedding(id)
        b = F.squeeze(self.b_embedding(id))
        # Reshape the vector to be the right dimensions for 2D conv
        W = F.reshape(W, (self.out_channels, self.in_channels, self.kh, self.kw))
        return F.convolution_2d(x, W, b, self.stride, self.pad)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    else:
        return x, x
