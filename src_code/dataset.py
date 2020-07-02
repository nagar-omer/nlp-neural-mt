from random import random

import torch
from torch.utils.data import Dataset


UNK = "</unk>"
SOS = "<s>"
EOS = "</s>"
PAD = "</p>"


class MTDataset(Dataset):
    def __init__(self, source_file, target_file, fixed_mapping=None, unk=0.):
        super(MTDataset, self).__init__()
        self._unk = unk
        self._build_data(source_file, target_file, fixed_mapping)

    @property
    def label_counts(self):
        return self._label_counts

    @property
    def len_source_vocab(self):
        return len(self._idx_to_ascii)

    @property
    def len_target_vocab(self):
        return len(self._idx_to_label)

    @property
    def target_start_index(self):
        return self._label_to_idx[SOS]

    @property
    def target_eof_index(self):
        return self._label_to_idx[EOS]

    @property
    def target_pad_index(self):
        return self._label_to_idx[PAD]

    @property
    def source_pad_index(self):
        return self._ascii_to_idx[PAD]

    @property
    def mapping(self):
        return self._label_to_idx, self._ascii_to_idx

    @staticmethod
    def _add_to_dict(dictionary, list_of_items):
        """
        add new items to a dictionary that maps item to index
        """
        for item in list_of_items:
            dictionary[item] = dictionary.get(item, len(dictionary))

    def _build_data(self, src, target, fixed_mapping=None):
        """
        this function builds
         - mapping from number to indices and vise versa
         - mapping from labels to indices and vise versa
         - data - pairs of number sequences and their corresponding labels (given as indices)
        """

        if fixed_mapping is None:
            label_to_idx = {UNK: 0, SOS: 1, EOS: 2, PAD: 3}
            ascii_to_idx = {UNK: 0, SOS: 1, EOS: 2, PAD: 3}
        else:
            label_to_idx, ascii_to_idx = fixed_mapping
        label_counts = {0: 1e-4, 1: 1e-4, 2: 1e-4, 3: 1e-4}
        data = []

        # loop pairs of source and destination sentences
        for src_sent, trg_sent in zip(open(src, "rt"), open(target, "rt")):

            # collect new numbers/labels
            src_sent, trg_sent = src_sent.split(), trg_sent.split()
            # if there is a fixed mapping - don't update it
            if fixed_mapping is None:
                self._add_to_dict(ascii_to_idx, src_sent)
                self._add_to_dict(label_to_idx, trg_sent)

            # count labels
            for label in trg_sent:
                idx = label_to_idx.get(label, label_to_idx[UNK])
                label_counts[idx] = label_counts.get(idx, 0) + 1
            label_counts[2] += 1  # EOF

            # convert to indices and save
            src_instance = [ascii_to_idx[SOS]] + [ascii_to_idx.get(num, ascii_to_idx[UNK]) for num in src_sent] + \
                           [ascii_to_idx[EOS]]
            trg_instance = [label_to_idx.get(label, label_to_idx[UNK]) for label in trg_sent] + [label_to_idx[EOS]]
            data.append((src_instance, trg_instance))

        self._label_counts = label_counts
        self._ascii_to_idx = ascii_to_idx
        self._label_to_idx = label_to_idx
        # obtain index to number/label mapping
        self._idx_to_label = [label for label, value in sorted(label_to_idx.items(), key=lambda x: x[1])]
        self._idx_to_ascii = [number for number, value in sorted(ascii_to_idx.items(), key=lambda x: x[1])]
        self._data = data

    @staticmethod
    def _pad(batch, dim, val, side="right"):
        """
        pad batch with zeros and convert to Long Tensor
        """
        if side == "right":
            padded = [torch.Tensor(b + [val]*(dim-len(b))) for b in batch]
        else:
            padded = [torch.Tensor([val] * (dim - len(b)) + b) for b in batch]
        return torch.stack(padded, dim=0).long()

    def _add_unk(self, batch):
        suorce_rand = self._ascii_to_idx[UNK]
        target_rand = self._label_to_idx[UNK]
        for source, target in batch:
            for i in range(len(target)):
                if random() < self._unk:
                    source[i+1] = suorce_rand
                    target[i] = target_rand
        return batch

    def collate_fn(self, batch):
        lengths = [len(b[0]) for b in batch]
        max_len_source = max(len(b[0]) for b in batch)
        max_len_target = max(len(b[1]) for b in batch)
        # batch = self._add_unk(batch)
        source_batch = MTDataset._pad([b[0] for b in batch], max_len_source, self._ascii_to_idx[PAD], side="left")
        target_batch = MTDataset._pad([b[1] for b in batch], max_len_target, self._label_to_idx[PAD], side="right")
        return source_batch, target_batch

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]


def sanity_test():
    from torch.utils.data import DataLoader
    ds = MTDataset("../data/dev.src", "../data/dev.trg")
    ds2 = MTDataset("../data/dev.src", "../data/dev.trg", fixed_mapping=ds.mapping, unk=0.15)

    dl = DataLoader(ds2,
                    num_workers=4,
                    batch_size=1,
                    collate_fn=ds2.collate_fn)
    for src_, dst_ in dl:
        # print(src_, lengths_, dst_)
        for s, t in zip(src_, dst_):
            print('{:<100}'.format(" ".join([ds._idx_to_ascii[i] for i in s])),
                  "\t", " ".join([ds._idx_to_label[i] for i in t]))


if __name__ == '__main__':
    sanity_test()
