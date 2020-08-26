import torch
import torch.utils.data as torch_data
import os
import data.utils

class dataset(torch_data.Dataset):

    def __init__(self, src, tgt, raw_src, raw_tgt):

        self.src = src
        self.tgt = tgt
        self.raw_src = raw_src
        self.raw_tgt = raw_tgt

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], \
               self.raw_src[index], self.raw_tgt[index]

    def __len__(self):
        return len(self.src)


class pg_dataset(torch_data.Dataset):

    def __init__(self, src, tgt, raw_src, raw_tgt, oovs):

        self.src = src
        self.tgt = tgt
        self.raw_src = raw_src
        self.raw_tgt = raw_tgt
        self.oovs = oovs

    def __getitem__(self, index):
        return self.src[index], self.tgt[index], \
               self.raw_src[index], self.raw_tgt[index], self.oovs[index]

    def __len__(self):
        return len(self.src)


def load_dataset(path):
    pass

def save_dataset(dataset, path):
    if not os.path.exists(path):
        os.mkdir(path)


def padding(data):
    #data.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt, raw_src, raw_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = s[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = s[:end]

    batch = {}
    batch['raw_src'] = raw_src
    batch['raw_tgt'] = raw_tgt
    batch['src'] = src_pad.t()
    batch['tgt'] = tgt_pad.t()
    batch['src_len'] = torch.LongTensor(src_len)
    batch['tgt_len'] = torch.LongTensor(tgt_len)

    return batch

def get_loader(dataset, batch_size, shuffle, num_workers, padding):

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=padding)
    return data_loader

def pg_padding(data):
    #data.sort(key=lambda x: len(x[0]), reverse=True)
    src, tgt, raw_src, raw_tgt, oovs = zip(*data)
    #print(tgt)
    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = s[:end]

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = s[:end]
    #tgt_len = [length-1 for length in tgt_len]
    num_oovs = max([len(x) for x in oovs])
    batch = {}
    batch['raw_src'] = raw_src
    batch['raw_tgt'] = raw_tgt
    batch['src'] = src_pad.t()
    batch['tgt'] = tgt_pad.t()
    batch['src_len'] = torch.LongTensor(src_len)
    batch['tgt_len'] = torch.LongTensor(tgt_len)
    batch['oovs'] = oovs
    batch['num_oovs'] = num_oovs

    #return src_pad.t(), src_len, tgt_pad.t(), tgt_len
    return batch