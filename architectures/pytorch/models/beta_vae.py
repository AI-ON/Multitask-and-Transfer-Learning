import torch
from torch import nn


class BetaVAE(nn.Module):
    def __init__(self, beta, encoder, decoder):
        super(BetaVAE, self).__init__()
        self.beta = beta
        self.encoder = encoder
        self.decoder = decoder

    def reparametrize(self, mu, logvar):
        '''Reshapes a unit-gaussian random matrix using the mu and
        logvar obtained from the encoder'''
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x.view(*self.in_shape))
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

    def get_loss_function(self):
        def loss_function(recon_x, x, mu, logvar):
            bce = nn.BCELoss(recon_x, x)
            kld_element = mu.pow(2).add_(logvar.exp()) \
                                   .mul_(-1).add_(1).add_(logvar)
            kld = torch.sum(kld_element).mul_(-0.5)
            return bce + self.beta * kld
