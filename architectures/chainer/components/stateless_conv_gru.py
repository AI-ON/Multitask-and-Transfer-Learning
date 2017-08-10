import numpy

import chainer.links as L
import chainer.functions as F
from chainer import Variable, Chain

class StatelessConvGRU2D(Chain):
    """Stateless Convolutional GRU

    Same calculation as a GRU, but with conv2d layers replacing FC
    layers.

    """
    def __init__(self, in_channels, out_channels,
                 ksize=None, stride=1, pad=0,
                 init=None, inner_init=None, bias_init=None):
        super(StatelessConvGRU2D, self).__init__()
        with self.init_scope():
            def make_W():
                return L.Convolution2D(
                    in_channels, out_channels,
                    ksize=ksize,
                    stride=stride,
                    pad=pad,
                    initialW=init,
                    initial_bias=bias_init,
                )

            def make_U():
                return L.Convolution2D(
                out_channels, out_channels,
                    ksize=ksize,
                    stride=stride,
                    pad=pad,
                    initialW=inner_init,
                    initial_bias=bias_init,
                )
            self.W_r = make_W()
            self.U_r = make_U()
            self.W_z = make_W()
            self.U_z = make_U()
            self.W = make_W()
            self.U = make_U()

    def __call__(self, x, h):
        z = self.W_z(x)
        h_bar = self.W(x)
        if h is not None:
            r = F.sigmoid(self.W_r(x) + self.U_r(h))
            z += self.U_z(h)
            h_bar += self.U(r * h)
        z = F.sigmoid(z)
        h_bar = F.tanh(h_bar)

        if h is not None:
            h_new = F.linear_interpolate(z, h_bar, h)
        else:
            h_new = z * h_bar
        return h_new
