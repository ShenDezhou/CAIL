import json
import os
import pandas

from CAIL2020.sfks.tools.dataset_tool import dfs_search

data_path = "../input/"
recursive = False
file_list = []
file_list = file_list + dfs_search(os.path.join(data_path, ''), recursive)
file_list = [file for file in file_list if 'train' in file]
file_list.sort()

rawinput = []
for filename in file_list:
    f = open(filename, "r", encoding='utf8')
    for line in f:
        data = json.loads(line)
        # filter dataset for Single option model and Multiple option model.
        # clean up answers.
        data["answer"] = [a for a in data["answer"] if a != "。"]
        rawinput.append(json.loads(line))

df = pandas.DataFrame(columns=["q","a","r"])
for item in rawinput:
    for option in list("ABCD"):
        x = dict(zip(list("qar"),(item['statement'], item['option_list'][option], option in item['answer'])))
        df = df.append(x, ignore_index=True)

df.to_csv('train.csv', encoding='utf-8', index=False)

