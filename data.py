from os.path import join

from torch.utils.data import Dataset
import torch


class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split, max_len):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.data_dir = data_dir
        self.split = split
        self.max_len = max_len
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = torch.load(join(self.data_dir, "{}.pkl".format(self.split)))
        for i, (img, formula) in enumerate(pairs):
            pair = (img, " ".join(formula.split()[:self.max_len]))
            pairs[i] = pair
        return pairs

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)
