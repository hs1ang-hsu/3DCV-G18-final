import json
import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
import csv
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from common.generators import ChunkedGenerator
from common.model import EmotionClassifier

def get_dataset(args, data, s):
    print(f"processing {s} data...")
    input = []
    gt = []
    for d in data:
        input.append(d['face_mesh'])
        gt.append(d['emotion'])
    return UnchunkedGenerator(input, gt, args.frame//2)

def main(args):
    data = np.load(args.dataset, allow_pickle=True)['data'].item()
    dataloader_train = get_dataset(args, data[TRAIN], TRAIN)
    dataloader_eval = get_dataset(args, data[VALID], VALID)
    
    
    
    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    dataloader_test = DataLoader(dataset, args.batch_size,
                    shuffle=False, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes
    ).to(args.device)
    model.eval()

    # load weights into model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)

    # TODO: predict dataset
    pred_data = []
    with torch.no_grad():
        for batch in dataloader_test:
            batch['input'] = batch['input'].to(args.device)
            batch['gt'] = batch['gt'].to(args.device)
            
            # eval
            res = model(batch)
            
            # result
            pred = res['pred_labels'].detach().cpu()
            for i, name in enumerate(batch['id']):
                pred_data.append([name, dataset.idx2label(pred[i].item())])
    
    pred_data.sort(key=lambda x: int(x[0][5:]))
    
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'intent'])
        for row in pred_data:
            writer.writerow(row)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name.",
        default="./dataset/data.npz",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
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

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
