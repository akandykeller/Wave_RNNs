import torch
import torchvision
import torchvision.transforms as transforms
import wandb
import numpy as np
import matplotlib.pyplot as plt

def get_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [57000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True,
                                               drop_last=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=bs_test,
                                              shuffle=False,
                                              drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False,
                                              drop_last=True)

    return train_loader, valid_loader, test_loader


def normalize_int(x):
  x -= x.min()
  x *= 255 / x.max()
  return x.int()

def normalize_int_np(x):
  x -= x.min()
  x *= 255 / x.max()
  return x.astype(np.int)


def Plot_Vid(seq, fps=60, vformat='gif', name='Latents', max_c=5):
    n_t, n_cin, n_hid = seq.shape
    # Seq shape should be T,C,H*W

    sqrt = int(np.ceil(np.sqrt(n_hid)))
    zeros = torch.zeros((seq.shape[0], 1, sqrt*sqrt))

    plot_c = min(n_cin, max_c)

    for c in range(plot_c):
      seq_norm = normalize_int(seq[:, c:c+1]).cpu()
      zeros[:, :, :seq.shape[2]] = seq_norm
      frame = zeros.view(-1, 1, sqrt, sqrt)
      wandb_video = wandb.Video(frame, fps=fps, format=vformat)
      wandb.log({name + f'_c_{c}': wandb_video})


def plot_conv_filters(weight, name, max_s=10, max_plots=3, max_inches=10, commit=False):
    weights_grid = weight.detach().cpu().numpy()
    empy_weight = np.zeros_like(weights_grid[0,0,:])
    c_out, c_in, h = weights_grid.shape
    sqrt_c = int(np.ceil(np.sqrt(c_out)))
    s = min(max_s, sqrt_c)

    for c in range(min(c_in, max_plots)):
      n_imgs = int(np.ceil(sqrt_c / s))
      for n in range(n_imgs):
          weights_grid[:, c] = normalize_int_np(weights_grid[:, c])
          f, axarr = plt.subplots(s,s)
          f.set_size_inches(min(s, max_inches), min(s, max_inches))

          for s_h in range(s):
              for s_w in range(s):
                  w_idx = s_h * s + s_w + n * s * s
                  if w_idx < c_out:
                      img = axarr[s_h, s_w].imshow(weights_grid[w_idx, c, :].reshape(1, -1, 1), cmap='PuBu_r')
                      axarr[s_h, s_w].get_xaxis().set_visible(False)
                      axarr[s_h, s_w].get_yaxis().set_visible(False)
                      # f.colorbar(img, ax=axarr[s_h, s_w])
                  else:
                      img = axarr[s_h, s_w].imshow(empy_weight.reshape(1, -1, 1), cmap='PuBu_r')
                      axarr[s_h, s_w].get_xaxis().set_visible(False)
                      axarr[s_h, s_w].get_yaxis().set_visible(False)
                      # f.colorbar(img, ax=axarr[s_h, s_w])

          wandb.log({"{}_cin{}_n{}".format(name, c, n): wandb.Image(plt)}, commit=commit)
          plt.close('all')


def plot_filters(weight, n_ch, name, max_s=10, max_inches=5, max_plots=10, commit=False):
    weights_grid = weight.detach().cpu().numpy()
    R, C = weights_grid.shape
    sz = max(R, C) / n_ch
    c_max = min(R, C)
    sqrt = int(np.ceil(np.sqrt(sz)))
    weights_grid = weights_grid.reshape(c_max, n_ch, sqrt, sqrt)
    
    empy_weight = np.zeros_like(weights_grid[0, 0])
    sqrt_c = int(np.ceil(np.sqrt(n_ch)))
    s = min(max_s, sqrt_c)

    for c in range(min(c_max, max_plots)):
      weights_grid[c] = normalize_int_np(weights_grid[c])
      n_imgs = int(np.ceil(sqrt_c / s))
      for n in range(n_imgs):
          f, axarr = plt.subplots(s,s)
          f.set_size_inches(min(s, max_inches), min(s, max_inches))

          for s_h in range(s):
              for s_w in range(s):
                  w_idx = s_h * s + s_w + n * s * s
                  if w_idx < n_ch:
                      img = axarr[s_h, s_w].imshow(weights_grid[c, w_idx], cmap='PuBu_r')
                      axarr[s_h, s_w].get_xaxis().set_visible(False)
                      axarr[s_h, s_w].get_yaxis().set_visible(False)
                  else:
                      img = axarr[s_h, s_w].imshow(empy_weight, cmap='PuBu_r')
                      axarr[s_h, s_w].get_xaxis().set_visible(False)
                      axarr[s_h, s_w].get_yaxis().set_visible(False)

          wandb.log({"{}_c{}_hid-ch{}".format(name, c, n): wandb.Image(plt)}, commit=commit)
          plt.close('all')