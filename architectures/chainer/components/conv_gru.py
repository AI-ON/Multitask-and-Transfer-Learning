import numpy

import chainer.links as L
import chainer.functions as F
from chainer import Variable, Chain


class ConvGRU2D(Chain):
    """Stateless Convolutional GRU

    Same calculation as a GRU, but with conv2d layers replacing FC
    layers.

    """
    def __init__(self, in_channels, out_channels,
                 ksize=None, stride=1, pad=0,
                 init=None, inner_init=None, bias_init=None):
        super(ConvGRU2D, self).__init__()
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
        self.reset_state()

    def to_cpu(self):
        super(Convolution2DGRU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(Convolution2DGRU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = F.sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)
        z = F.sigmoid(z)
        h_bar = F.tanh(h_bar)

        if self.h is not None:
            h_new = F.linear_interpolate(z, h_bar, self.h)
        else:
            h_new = z * h_bar
        self.h = h_new  # save the state
        return h_new
