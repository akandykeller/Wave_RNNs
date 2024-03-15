import torch
import torch.nn.functional as F
import numpy as np
import wandb
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

def str_to_bool(s):
    if s.lower() == 'true' or s.lower() == 'yes':
         return True
    elif s.lower() == 'false' or s.lower() == 'no':
         return False
    else:
         raise ValueError 

def get_batch(num_samples, sample_len, one_hot='True', memory_len=10):
    # assert(sample_len > 20)
    X = np.zeros((num_samples, sample_len + memory_len))
    data = np.random.randint(low = 1, high = 9, size = (num_samples, memory_len))
    X[:, :memory_len] = data
    X[:, -(memory_len + 1)] = 9
    Y = np.zeros((num_samples, sample_len + memory_len))
    Y[:, -memory_len:] = X[:, :memory_len]

    if str_to_bool(one_hot) == True:
        return F.one_hot(torch.tensor(X, dtype=torch.int64).permute(1,0), 10).float(), torch.tensor(Y).int().permute(1,0)
    else:
        return torch.tensor(X).float().permute(1,0), torch.tensor(Y).int().permute(1, 0)


def normalize_int(x):
  x -= x.min()
  x *= 255 / x.max()
  return x.int()

def normalize_int_np(x):
  x -= x.min()
  x *= 255.0 / x.max()
  return x.astype(np.int)


def Plot_Vid(seq, fps=60, vformat='gif', name='Latents', max_c=3):
    n_t, n_cin, nh, nw = seq.shape
    # Seq shape should be T,C,H,W

    seq_norm = normalize_int(seq).cpu()
    
    for c in range(min(n_cin, max_c)):
        wandb_video = wandb.Video(seq_norm[:, c].unsqueeze(1), fps=fps, format=vformat)
        wandb.log({name + f'c_{c}': wandb_video})


def plot_output(data, output, label):
    """Shape = (T, n_inp)"""
    data = data.detach().T.cpu().numpy()
    output = F.softmax(output, dim=-1).detach().T.cpu().numpy()
    label = F.one_hot(label.type(torch.int64), 10).float().detach().T.cpu().numpy()

    if data.shape[1] > 50:
        data = data[:, :50]
        output = output[:, -50:]
        label = label[:, -50:]
        title_postfix = '_50-sample'
    else:
        title_postfix = '_all'
    
    f, axs = plt.subplots(3)
    f.set_size_inches(10, 6)

    img1 = axs[0].imshow(data, cmap='PuBu_r')
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)
    axs[0].set_title('Input' + title_postfix)
    img2 = axs[1].imshow(output, cmap='PuBu_r')
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)
    axs[1].set_title('Output' + title_postfix)
    img2 = axs[2].imshow(label, cmap='PuBu_r')
    axs[2].get_xaxis().set_visible(False)
    axs[2].get_yaxis().set_visible(False)
    axs[2].set_title('Label' + title_postfix)

    wandb.log({"Output" : wandb.Image(plt)}, commit=True)
    plt.close('all')