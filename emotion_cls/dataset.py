from typing import List, Dict
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    def __init__(
        self,
        data: List[Dict]
    ):
        input = []
        gt = []
        print("processing data...")
        for d in data:
            input.append(d['face_mesh'])
            gt.append(d['emotion'])
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def collate_fn(self, samples: List[Dict]) -> Dict:
        batch = {}
        batch['input'] = torch.tensor([s['input'] for s in samples])
        batch['gt'] = torch.tensor([s['gt'] for s in samples])
        
        return batch