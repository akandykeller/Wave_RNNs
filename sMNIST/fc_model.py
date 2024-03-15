from torch import nn
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, device, n_ch=1, act='tanh', ksize=3, mlp_decoder='no', init='eye', freeze_rnn='no', freeze_encoder='no', solo_init='no'):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.spatial = int(np.sqrt(n_hid))
        self.mlp_decoder = mlp_decoder
        if mlp_decoder == 'yes':
            self.readout = nn.Sequential(
                    nn.Linear(784, n_hid),
                    nn.ReLU(),
                    nn.Linear(n_hid, n_out)
                    )
        else:
            assert mlp_decoder == 'no'
            self.readout = nn.Linear(784, n_out)

    def forward(self, x, get_seq=False):
        ## initialize hidden states
        y_seq = []

        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)

        output = self.readout(x)

        return output, y_seq

    def plot_weights(self):
        pass