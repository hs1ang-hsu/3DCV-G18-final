import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from time import time
from tqdm import tqdm
import os
import sys
import numpy as np

import torch
from torch.utils.data import DataLoader

from common.generators import ChunkedGenerator
from common.model import EmotionClassifier

TRAIN = 'train'
VALID = 'validation'

def train(model, dataloader, optimizer, args):
    correct_train = 0
    N = 0
    progress = tqdm(total=dataloader.num_batches)
    for batch_mesh, batch_emotion in dataloader.next_epoch():
        inputs_mesh = torch.from_numpy(batch_mesh.astype('float32'))
        inputs_emotion = torch.from_numpy(batch_emotion.astype('float32'))
        inputs_mesh = inputs_mesh.to(args.device)
        inputs_emotion = inputs_emotion.to(args.device)
        optimizer.zero_grad()
        
        # train
        pred, loss = model(inputs_mesh, inputs_emotion)
        loss.backward()
        optimizer.step()
        
        # acc
        pred = pred.detach().cpu()
        gt = inputs_emotion.detach().cpu()
        
        correct_train += (pred == gt.view_as(pred)).sum().item()
        N += gt.size(0)
        progress.update(1)
    
    return correct_train / N

def evaluate(model_train_dict, model_eval, dataloader_eval, args):
    correct_eval = 0
    N = 0
    progress = tqdm(total=dataloader_eval.num_batches)
    with torch.no_grad():
        model_eval.load_state_dict(model_train_dict)
        model_eval.eval()
        for batch_mesh, batch_emotion in dataloader_eval.next_epoch():
            inputs_mesh = torch.from_numpy(batch_mesh.astype('float32'))
            inputs_emotion = torch.from_numpy(batch_emotion.astype('float32'))
            inputs_mesh = inputs_mesh.to(args.device)
            inputs_emotion = inputs_emotion.to(args.device)
            
            # evaluate
            pred, loss = model_eval(inputs_mesh, inputs_emotion)
            
            # acc
            pred = pred.detach().cpu()
            gt = inputs_emotion.detach().cpu()
            
            correct_eval += (pred == gt.view_as(pred)).sum().item()
            N += gt.size(0)
            progress.update(1)
    return correct_eval / N

def get_dataset(args, data, s):
    print(f"processing {s} data...")
    input = []
    gt = []
    for d in data:
        input.append(d['face_mesh'])
        gt.append(d['emotion'])
    return ChunkedGenerator(args.batch_size, input, gt, args.frame//2)


def main(args):
    data = np.load(args.dataset, allow_pickle=True)['data'].item()
    dataloader_train = get_dataset(args, data[TRAIN], TRAIN)
    dataloader_eval = get_dataset(args, data[VALID], VALID)
    
    model = EmotionClassifier(args.kp, args.feature_dim, args.hidden_dim, args.channels,
                    args.out_dim, args.num_classes, args.using_trans).to(args.device)
    model_eval = EmotionClassifier(args.kp, args.feature_dim, args.hidden_dim, args.channels,
                    args.out_dim, args.num_classes, args.using_trans).to(args.device)
    
    # number of params
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('Trainable parameter count:', model_params)

    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)

    acc_history_train = []
    acc_history_val = []
    lr = args.lr
    best_acc = 0.
    for epoch in range(args.num_epoch):
        start_time = time()
        model.train()
        
        # Training loop
        acc_train = train(model, dataloader_train, optimizer, args)
        acc_history_train.append(acc_train)
        
        # Evaluation loop
        acc_eval = evaluate(model.state_dict(), model_eval, dataloader_eval, args)
        acc_history_val.append(acc_eval)
        
        # Saving data
        elapsed = (time() - start_time) / 60
        print('[%d] time %.2f lr %f train %f eval %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    acc_train,
                    acc_eval))
        
        if acc_eval >= best_acc:
            best_acc = acc_eval
            chk_path = os.path.join(args.checkpoint_dir, 'best.bin')
            print('Saving best checkpoint to', chk_path)
            torch.save(model.state_dict(), chk_path)
        
        # update params
        lr *= args.lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lrd
        
        chk_path = os.path.join(args.checkpoint_dir, 'final.bin')
        torch.save(model.state_dict(), chk_path)
        
        if args.export_training_curves and epoch > 1:
            if 'matplotlib' not in sys.modules:
                import matplotlib

                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(1, len(acc_history_train)) + 1
            plt.plot(epoch_x, acc_history_train[1:], '--', color='C0')
            plt.plot(epoch_x, acc_history_val[1:], '--', color='C1')
            plt.legend(['train', 'eval'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.xlim((1, epoch+1))
            plt.savefig(os.path.join(args.checkpoint_dir, 'acc.png'))
            plt.close('all')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name.",
        default="./dataset/data.npz",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./checkpoints/",
    )
    
    # model
    parser.add_argument("--frame", type=int, default=27)
    parser.add_argument("--kp", type=int, default=34)
    parser.add_argument("--feature_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--out_dim", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument('--using_trans', action='store_true')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lrd", type=float, default=0.95)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--export_training_curves", type=bool, default=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    main(args)
