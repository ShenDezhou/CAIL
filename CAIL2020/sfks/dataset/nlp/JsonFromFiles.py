import json
import os
from torch.utils.data import Dataset
import random

from tools.dataset_tool import dfs_search

from gbt.SingleMulti import SingleMulti


class JsonFromFilesDataset(Dataset):
    def __init__(self, config, mode,  encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = config.get("data", "%s_data_path" % mode)
        self.encoding = encoding
        self.siglemulti = SingleMulti('gbt/statement_tfidf.model', 'gbt/statement_som_gbt.model')

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = False

        multi = config.getboolean("data", "multi_choice")


        for name in filename_list:
            self.file_list = self.file_list + dfs_search(os.path.join(self.data_path, name), recursive)
        self.file_list.sort()

        self.data = []
        for filename in self.file_list:
            f = open(filename, "r", encoding=encoding)
            for line in f:
                data = json.loads(line)
                if mode == "test":
                    self.data.append(json.loads(line))
                    continue

                statementoption = data['statement']
                for op in data["option_list"].values():
                    statementoption += "。"
                    statementoption += op

                aimodel = self.siglemulti.checkSingleMulti(statementoption)
                # filter dataset for Single option model and Multiple option model.
                data["answer"] = [a for a in data["answer"] if a != "。"]  #clean up answers.

                if multi:
                    if aimodel:
                        if len(data["answer"]) > 1:
                            self.data.append(json.loads(line))
                        # else:
                        #     if random.randint(0, 2) > 0:
                        #         self.data.append(json.loads(line))

                else:
                    if not aimodel and len(data["answer"]) == 1:
                        self.data.append(json.loads(line))

                # if (not multi) and len(data["answer"]) != 1:
                #     if mode != "test":
                #         continue


        if mode == "train":
            random.shuffle(self.data)

        self.reduce = config.getboolean("data", "reduce")
        if mode != "train":
            self.reduce = False
        if self.reduce:
            self.reduce_ratio = config.getfloat("data", "reduce_ratio")

    def __getitem__(self, item):
        if self.reduce:
            return self.data[random.randint(0, len(self.data) - 1)]
        return self.data[item]

    def __len__(self):
        if self.reduce:
            return int(self.reduce_ratio * len(self.data))
        return len(self.data)
