from torch import nn, optim
import torch
import rnn_model as model_rnn
import conv_rnn_model as model_conv_rnn
import fc_model as model_fc
import loco_model
import torch.nn.utils
import utils
from pathlib import Path
import argparse
from tqdm import tqdm
import os
import gc
from sys import exit

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--run_name', type=str, default=None,
                    help='name of run for wandb')
parser.add_argument('--dataset', type=str, default='sMNIST',
                    help='Just used for weights and biases logging, does not affect training')
parser.add_argument('--model_type', type=str, default='rnn',
                    help='type of model, rnn, wrnn, fc, loco')
parser.add_argument('--n_hid', type=int, default=256,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=120,
                    help='max epochs')
parser.add_argument('--batch', type=int, default=128,
                    help='batch size')
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
parser.add_argument('--act', type=str, default='tanh',
                    help='hidden state activation')
parser.add_argument('--init', type=str, default='eye',
                    help='initalization for RNN')
parser.add_argument('--freeze_rnn', type=str, default='no',
                    help='Make Recurrent weights untrained')
parser.add_argument('--freeze_encoder', type=str, default='no',
                    help='Make Encoder weights untrained')
parser.add_argument('--mlp_decoder', type=str, default='no')
parser.add_argument('--solo_init', type=str, default='no')
parser.add_argument('--grad_clip', type=float, default=0.0)


args = parser.parse_args()
print(args)

n_inp = 1
n_out = 10
bs_test = 1000

device='cuda'

model_select = {'rnn': model_rnn, 'wrnn': model_conv_rnn,
                'fc': model_fc, 'loco': loco_model}

model = model_select[args.model_type].coRNN(n_inp, args.n_hid, n_out,device,
                                            args.n_ch, args.act, args.ksize, args.mlp_decoder,
                                            args.init, args.freeze_rnn, args.freeze_encoder,
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

objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

fname = f'result/sMNIST_log_{args.model_type}_h{args.n_hid}_lr{args.lr}.txt'

def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(data_loader, desc="Test")):
            images = images.reshape(bs_test, 1, 784)
            images = images.permute(2, 0, 1).to(device)
            labels = labels.to(device)

            output, _  = model(images)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

# checkpoint = torch.load(LOAD_PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# args.lr = checkpoint['lr']
start_epoch = 0

for epoch in range(start_epoch, args.epochs):
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        images = images.reshape(-1, 1, 784)
        images = images.permute(2, 0, 1).to(device)
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
            if len(y_seq) > 0:
                utils.Plot_Vid(y_seq.detach().cpu()[:, 0])
                utils.Plot_Seq(y_seq.detach().cpu()[:, 0])
                utils.Plot_FFT(y_seq.detach().cpu()[:, 0])

            # del y_seq
            gc.collect()
            if torch.isnan(loss):
                exit()


    valid_acc = test(valid_loader)
    test_acc = test(test_loader)
    log('Valid Acc:', valid_acc)
    log('Test Acc:', test_acc)


    Path('result').mkdir(parents=True, exist_ok=True)
    f = open(fname, 'a')
    f.write('eval accuracy: ' + str(round(valid_acc, 2)) + '\n')
    f.close()

    if (epoch+1) % args.lr_drop_epoch == 0:
        args.lr /= float(args.lr_drop_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': args.lr,
            'loss': loss,
            }, SAVE_PATH)

log('Final Test Acc:', test_acc)
f = open(fname, 'a')
f.write('final test accuracy: ' + str(round(test_acc, 2)) + '\n')
f.close()
