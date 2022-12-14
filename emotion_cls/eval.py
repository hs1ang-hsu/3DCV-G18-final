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
from collections import Counter

import torch
from torch.utils.data import DataLoader

from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.model import EmotionClassifier

TRAIN = 'train'
VALID = 'validation'

def eval_data_prepare(receptive_field, inputs_mesh):
    inputs_mesh = torch.squeeze(inputs_mesh)
    out_num = inputs_mesh.shape[0] - receptive_field + 1
    eval_input_mesh = torch.empty(out_num, receptive_field, inputs_mesh.shape[1], inputs_mesh.shape[2])
    for i in range(out_num):
        eval_input_mesh[i,:,:,:] = inputs_mesh[i:i+receptive_field, :, :]
    return eval_input_mesh

def evaluate_frame(model_eval, dataloader_eval, args):
    correct_eval = 0
    N = 0
    progress = tqdm(total=dataloader_eval.num_batches)
    with torch.no_grad():
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

def evaluate(model_eval, dataloader_eval, args):
    correct_eval = 0
    N = 0
    progress = tqdm(total=dataloader_eval.num_batches)
    with torch.no_grad():
        model_eval.eval()
        for batch_mesh, batch_emotion in dataloader_eval.next_epoch():
            inputs_mesh = eval_data_prepare(args.frame, torch.from_numpy(batch_mesh.astype('float32')))
            inputs_emotion = torch.from_numpy(batch_emotion.astype('float32'))
            
            inputs_mesh = inputs_mesh.to(args.device)
            inputs_emotion = inputs_emotion.to(args.device)
            
            # evaluate
            pred, loss = model_eval(inputs_mesh, inputs_emotion)
            
            # acc
            pred = pred.detach().cpu()
            gt = inputs_emotion.detach().cpu()
            
            counter = Counter(pred)
            most_common = counter.most_common(1)[0][0]
            correct_eval += 1 if most_common==gt[0][0] else 0
            N += 1
            
            progress.update(1)
    return correct_eval / N

def get_dataset(args, data, s):
    print(f"processing {s} data...")
    input = []
    gt = []
    for d in data:
        input.append(d['face_mesh'])
        gt.append(d['emotion'])
    # return UnchunkedGenerator(input, gt, args.frame//2)
    return ChunkedGenerator(64, input, gt, args.frame//2)


def main(args):
    data = np.load(args.dataset, allow_pickle=True)['data'].item()
    dataloader_train = get_dataset(args, data[TRAIN], TRAIN)
    dataloader_eval = get_dataset(args, data[VALID], VALID)
    
    model = EmotionClassifier(args.kp, args.feature_dim, args.hidden_dim, args.channels,
                    args.out_dim, args.num_classes, args.using_trans).to(args.device)
    
    checkpoint = torch.load(args.evaluate)
    model.load_state_dict(checkpoint)
        
    # Training loop
    # acc_train = evaluate(model, dataloader_train, args)
    acc_train = evaluate_frame(model, dataloader_train, args)
    
    # Evaluation loop
    # acc_eval = evaluate(model, dataloader_eval, args)
    acc_eval = evaluate_frame(model, dataloader_eval, args)
    
    # Saving data
    print('train: %f, eval: %f' % (acc_train, acc_eval))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name.",
        default="./dataset/data.npz",
    )
    parser.add_argument(
        "--evaluate",
        type=Path,
        help="Directory to the model file.",
        default="./checkpoints/best.bin",
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
    main(args)
