import json
import os

import numpy
import thulac
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import re

#training score : 0.673

def cut_text(alltext):
    count = 0
    cut = thulac.thulac(seg_only=True)
    train_text = []
    for text in alltext:
        count += 1
        if count % 2000 == 0:
            print(count)
        train_text.append(cut.cut(text, text=True))

    return train_text


def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=5,
        max_features=5000,
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1
    )
    tfidf.fit(train_data)

    return tfidf

def print_mem():
	process = psutil.Process(os.getpid())  # os.getpid()
	memInfo = process.memory_info()
	return '{:.4f}G'.format(1.0 * memInfo.rss / 1024 /1024 /1024)


data_path="../data"
filename_list="sfzy_small.json".replace(" ", "").split(",")

pattern = '(.{2,4})纠纷一案'
pr = re.compile(pattern)
statement =[]
target=[]
reasondic = dict(zip(['劳动合同', '侵权责任', '租赁合同', '借款合同', '继承', '追偿权', '借款', '侵权', '继承关系'], range(9)))


for filename in filename_list:
    f = open(os.path.join(data_path, filename), "r", encoding='utf-8')
    for line in f:
        data = json.loads(line)
        text = data['summary']
        summary0 = text.split('。')[0]
        reason = pr.findall(summary0)[0]

        text = data['text']
        for dic in text:
            if pr.search(dic['sentence']):
                new_reason = pr.findall(dic['sentence'])[0]
                break
