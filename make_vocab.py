from os.path import join
import pickle as pkl


START_TOKEN = 0
PAD_TOKEN = 1
END_TOKEN = 2
UNK_TOKEN = 3

# buid sign2id


class Vocab(object):
    def __init__(self):
        self.sign2id = {"<s>": START_TOKEN, "</s>": END_TOKEN,
                        "<pad>": PAD_TOKEN, "<unk>": UNK_TOKEN}
        self.id2sign = dict((idx, token)
                            for token, idx in self.sign2id.items())
        self.length = 4

    def add_sign(self, sign):
        if sign not in self.sign2id:
            self.sign2id[sign] = self.length
            self.id2sign[self.length] = sign
            self.length += 1

    def add_formula(self, formula):
        for sign in formula:
            self.add_sign(sign)

    def __len__(self):
        return self.length


def make_vocab(data_dir):
    """
    traverse training formulas to make vocab
    and store the vocab in the file
    """
    vocab = Vocab()

    formulas_file = join(data_dir, 'formulas.norm.lst')
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    with open(join(data_dir, 'train_filter.lst'), 'r') as f:
        for line in f:
            _, idx = line.strip('\n').split()
            idx = int(idx)
            formula = formulas[idx].split()
            vocab.add_formula(formula)

    vocab_file = join(data_dir, 'vocab.pkl')
    print("Writing Vocab File in ", vocab_file)
    with open(vocab_file, 'wb') as w:
        pkl.dump(vocab, w)
