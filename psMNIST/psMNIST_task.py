from torch import nn, optim
import torch
import rnn_model as model_rnn
import conv_rnn_model as model_conv_rnn
import torch.nn.utils
import utils
from pathlib import Path
import argparse
from tqdm import tqdm
import os
from sys import exit

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--run_name', type=str, default=None,
                    help='name of run for wandb')
parser.add_argument('--model_type', type=str, default='rnn',
                    help='type of model, rnn or wrnn')
parser.add_argument('--dataset', type=str, default='psMNIST',
                    help='')
parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr_scheduler', type=str, default='step')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--lr_drop_rate', type=float, default=10.0,
                    help='learning rate')
parser.add_argument('--lr_drop_epoch', type=int, default=100,
                    help='learning rate')
parser.add_argument('--n_ch', type=int, default=1,
                    help='Num hidden state channels')
parser.add_argument('--ksize', type=int, default=3,
                    help='Hidden Kernelsize')
parser.add_argument('--act', type=str, default='relu',
                    help='hidden state activation')
parser.add_argument('--init', type=str, default='eye',
                    help='initalization for RNN')
parser.add_argument('--freeze_rnn', type=str, default='no',
                    help='Make Recurrent weights untrained')
parser.add_argument('--mlp_decoder', action='store_true')
parser.add_argument('--grad_clip', type=float, default=0.0)
parser.add_argument('--solo_init', type=str, default='no')

args = parser.parse_args()
print(args)

# torch.manual_seed(12008)
n_inp = 1
n_out = 10
bs_test = 1000

device='cuda'

perm = torch.randperm(784)


model_select = {'rnn': model_rnn, 'wrnn': model_conv_rnn}


model = model_select[args.model_type].coRNN(n_inp, args.n_hid, n_out, device,
                                            args.n_ch, args.act, args.ksize, args.mlp_decoder,
                                            args.init, args.freeze_rnn,
                                            args.solo_init).to(device)
train_loader, valid_loader, test_loader = utils.get_data(args.batch,bs_test)

import wandb
wandb.init(name=args.run_name,
            project='PROJECT_NAME', 
            entity='ENTITY_NAME', 
            dir='WANDB_DIR',
            config=args)            
wandb.watch(model)

SAVE_PATH = os.path.join(wandb.run.dir, 'checkpoint.tar')

def log(key, val):
    print(f"{key}: {val}")
    wandb.log({key: val})

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

log('n_params', count_parameters(model))

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

if args.lr_scheduler == 'onecycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
elif args.lr_scheduler == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=1.0/args.lr_drop_rate, patience=5, cooldown=5, verbose=True)
elif args.lr_scheduler == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0, verbose=True)
elif args.lr_scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_epoch, gamma=1.0/args.lr_drop_rate)
elif args.lr_scheduler == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
else:
    raise NotImplementedError

fname = f'result/psMNIST_log_{args.model_type}_h{args.n_hid}_lr{args.lr}.txt'

def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.reshape(bs_test, 1, 784)
            images = images.permute(2, 0, 1)
            images = images[perm, :, :].to(device)
            labels = labels.to(device)

            output, _ = model(images, get_seq=False)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

# checkpoint = torch.load(SAVE_PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# args.lr = checkpoint['lr'


best_eval = 0.
for epoch in range(args.epochs):
    log('Epoch', epoch)
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        images = images.reshape(-1, 1, 784)
        images = images.permute(2, 0, 1)
        images = images[perm, :, :].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output, _ = model(images, get_seq=False)
        loss = objective(output, labels)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if i % 100 == 0:
            log('Train Loss:', loss)
            _, y_seq = model(images, get_seq=True)
            utils.Plot_Vid(y_seq.detach().cpu()[:, 0])
            if torch.isnan(loss):
                exit()

    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    log('Valid Acc:', valid_acc)
    log('Test Acc:', test_acc)

    if(valid_acc > best_eval):
        best_eval = valid_acc
        final_test_acc = test_acc

    Path('result').mkdir(parents=True, exist_ok=True)
    f = open(fname, 'a')
    f.write('eval accuracy: ' + str(round(valid_acc,2)) + '\n')
    f.close()

    if args.lr_scheduler != 'plateau':
        scheduler.step()
    else:
        scheduler.step(valid_acc)
    current_lr = scheduler.optimizer.param_groups[0]['lr']
    log('lr', current_lr)

    # if (epoch+1) % args.lr_drop_epoch == 0:
        # args.lr /= float(args.lr_drop_rate)
        # for param_group in optimizer.param_groups:
            # param_group['lr'] = args.lr

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': args.lr,
            'loss': loss,
            }, SAVE_PATH)

log('Final Test Acc:', test_acc)
f = open(fname, 'a')
f.write('final test accuracy: ' + str(round(final_test_acc, 2)) + '\n')
f.close()

