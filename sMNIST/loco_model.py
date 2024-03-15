from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LocallyConnected1d(nn.Module):
    """
    Adapted from: https://discuss.pytorch.org/t/locally-connected-layers/26979
    """
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding=0, padding_mode='constant', bias=True):
        super(LocallyConnected1d, self).__init__()

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], kernel_size[0])
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = kernel_size[0]
        self.stride = stride
        self.padder = lambda x: F.pad(x, (padding, padding), mode=padding_mode)
        
        torch.nn.init.xavier_uniform_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        _, c, h = x.size()
        kh = self.kernel_size
        dh = self.stride

        x = self.padder(x).unfold(2, kh, dh)
        x = x.contiguous().view(*x.size()[:-1], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class RNN_Cell(nn.Module):
    def __init__(self, n_inp, n_hid, n_ch=1, act='tanh', ksize=3, init='eye', freeze_rnn='no', freeze_encoder='no', solo_init='no'):
        super(RNN_Cell, self).__init__()
        self.n_hid = n_hid
        self.n_ch = n_ch
        self.Wx = nn.Linear(n_inp, n_hid * n_ch)
        self.Wy = LocallyConnected1d(n_ch, n_ch, output_size=[n_hid], kernel_size=[ksize], stride=1, padding=ksize//2, padding_mode='circular')

        if solo_init == 'yes':
            nn.init.zeros_(self.Wx.weight)
            nn.init.zeros_(self.Wx.bias)
            with torch.no_grad():
                w = self.Wx.weight.view(n_ch, n_hid, n_inp)
                w[:, 0] = 1.0
        elif solo_init == 'no':
            nn.init.normal_(self.Wx.weight, mean=0.0, std=0.001)
        else:
            raise NotImplementedError

        if act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'ident':
            self.act = nn.Identity()
        else:
            raise NotImplementedError
        
        assert init in ['eye', 'fwd', 'rand']
        
        if init == 'eye':
            wts = torch.zeros(n_ch, n_ch, ksize)
            nn.init.dirac_(wts)
            wts = wts.unsqueeze(2).repeat(1, 1, n_hid, 1).unsqueeze(0)

        elif init == 'fwd':
            wts = torch.zeros(n_ch, n_ch, ksize)
            nn.init.dirac_(wts)
            wts = torch.roll(wts, 1, -1)
            wts = wts.unsqueeze(2).repeat(1, 1, n_hid, 1).unsqueeze(0)
            # wts = wts * (1.0 - wave_speed) + torch.roll(wts, 1, -1) * wave_speed

        if init == 'eye' or init == 'fwd':
            with torch.no_grad():
                self.Wy.weight.copy_(wts)

        if freeze_encoder == 'yes':
            for param in self.Wx.parameters():
                param.requires_grad = False
        else:
            assert freeze_encoder == 'no'

        if freeze_rnn == 'yes':
            for param in self.Wy.parameters():
                param.requires_grad = False
        else:
            assert freeze_rnn == 'no'


    def forward(self,x,hy):
        hy = self.act(self.Wx(x) + self.Wy(hy.view(-1, self.n_ch, self.n_hid)).flatten(start_dim=1))
        return hy

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, device, n_ch=1, act='tanh', ksize=3, mlp_decoder='no', init='eye', freeze_rnn='no', freeze_encoder='no', solo_init='no'):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.n_ch = n_ch
        self.spatial = int(np.sqrt(n_hid))
        self.mlp_decoder = mlp_decoder
        self.cell = RNN_Cell(n_inp, n_hid, n_ch, act, ksize, init, freeze_rnn, freeze_encoder, solo_init)
        if mlp_decoder == 'yes':
            self.readout = nn.Sequential(
                    nn.Linear(self.n_hid * self.n_ch, 100),
                    nn.ReLU(),
                    nn.Linear(100, n_out)
                    )
        else:
            assert mlp_decoder == 'no'
            self.readout = nn.Linear(self.n_hid * self.n_ch, n_out)

    def forward(self, x, get_seq=False):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1), self.n_hid * self.n_ch)).to(device)
        y_seq = []

        for t in range(x.size(0)):
            hy = self.cell(x[t], hy)
            if get_seq:
                y_seq.append(hy.view(x.size(1), self.n_ch, -1).detach().cpu())
        output = self.readout(hy)

        if get_seq:
            y_seq = torch.stack(y_seq, dim=0)

        return output, y_seq

    def plot_weights(self):
        pass

