import json

# files = ["../data/train19_dev.json", "../data/train19_train.json", "../data/train19_test.json", "train_large.json"]
files = ["../data/train19_dev.json",  "../data/train19_test.json", "train_small.json",  "train_large.json"]

def read(file, start_id=0):
    with open(file, 'r', encoding='utf-8') as f:
        train = json.load(f)
        trainx = []
        for dic in train:
            dic['_id'] = start_id
            trainx.append(dic)
            start_id += 1
    return trainx

result = []
with open('data2/train_full.json', 'w', encoding='utf-8') as fw:
    startid = 0
    for file in files:
        fdic = read(file, start_id=startid)
        startid += len(fdic)
        result.extend(fdic)
    json.dump(result, fw, indent=4, ensure_ascii=False)
print('FIN')