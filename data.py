from os.path import join

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class Im2LatexDataset(Dataset):
    def __init__(self, data_dir, split, transform=transforms.ToTensor()):
        """args:
        data_dir: root dir storing the prepoccessed data
        split: train, validate or test
        """
        assert split in ["train", "validate", "test"]
        self.data_dir = data_dir
        self.images_dir = join(data_dir, "images_processed")
        self.formulas = self._get_formulas()
        self.transform = transform
        self.pairs = self._get_pairs(split)

    def __getitem__(self, index):
        return self.pairs[index]

    def __len__(self):
        return len(self.pairs)

    def _get_formulas(self):
        formulas_file = join(self.data_dir, "formulas.norm.lst")
        with open(formulas_file, 'r') as f:
            formulas = [formula.strip('\n') for formula in f.readlines()]
        return formulas

    def _get_pairs(self, split):
        # the line in this file map image to formulas
        map_file = join(self.data_dir, split + "_filter.lst")
        # get image-formulas pairs
        pairs = []
        with open(map_file, 'r') as f:
            for line in f:
                img_name, formula_id = line.strip('\n').split()
                # load img and its corresponding formula
                img_path = join(self.images_dir, img_name)
                img = Image.open(img_path)
                img_tensor = self.transform(img)
                formula = self.formulas[int(formula_id)]
                pair = (img_tensor, formula)
                pairs.append(pair)
        pairs.sort(key=img_size)
        return pairs


def img_size(pair):
    img, formula = pair
    return tuple(img.size())
