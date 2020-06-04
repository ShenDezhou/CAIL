import json
import os

import numpy
import thulac
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import re

#training score :  0.842
cut = thulac.thulac(seg_only=True)
types=5
def cut_text(alltext):
    count = 0
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


def score(candidate, summarypool):
    target_data = cut_text([candidate])
    target_set = set(target_data[0].split(' '))
    # print(target_set)
    score = []
    for j in range(len(summarypool)):
        score.append(len(summarypool[j] & target_set))

    return max(score)


data_path="../data"
filename_list="sfzy_small.json".replace(" ", "").split(",")

patterns = "判决如下"
ps = re.compile(patterns)
patterne1 = '^案件受理费'  # 如不(服)?本判决
patterne2 = '如不(服)?本判决'
pe1 = re.compile(patterne1)
pe2 = re.compile(patterne2)

statement =[]
target=[]


for filename in filename_list:
    f = open(os.path.join(data_path, filename), "r", encoding='utf-8')
    for line in f:
        data = json.loads(line)
        text = data['text']
        summary = data['summary']

        candidate = []
        sumlines = summary.split("。")
        sumlines = [line for line in sumlines if len(line) > 0]
        train_data = cut_text(sumlines)
        for line in train_data:
            candidate.append(set(line.split(' ')))

        tap = False
        for dic in text:
            if tap and (pe1.search(dic['sentence']) or pe2.search(dic['sentence'])):
                tap = False

            if tap:
                ss = score(dic['sentence'], candidate)
                if ss > 10:
                    statement.append(dic['sentence'])
                    target.append(1)
                else:
                    statement.append(dic['sentence'])
                    target.append(0)

            if ps.search(dic['sentence']):
                tap = True



print('n:', len(statement),"classes:", target)
train_data = cut_text(statement)
joblib.dump(train_data,'statement_seg.model', compress=3)
train_data = joblib.load('statement_seg.model')

print('train tfidf...', print_mem())
tfidf = train_tfidf(train_data)
joblib.dump(tfidf,'statement_tfidf.model', compress=3)
vec = tfidf.transform(train_data)
numpy.savez_compressed("statement_vec.npz", vec.todense())

print('train gbt...', print_mem())
gbt = GradientBoostingClassifier(learning_rate=0.01,
                                 n_estimators=100,
                                 max_depth=10,
                                 min_samples_leaf =10,
                                 min_samples_split =20,
                                 # max_features=9,
                                 verbose=1,
                                 ).fit(vec, target)
joblib.dump(gbt, 'statement_som_gbt.model')

yp = gbt.predict(vec)
print("training score : %.3f " % gbt.score(vec, target))
# correct=0
# wrong=0
# for i in range(len(yp)):
#     if target[i]==0 and yp[i]==0:
#         correct += 1
#     elif target[i] == 1 and yp[i] == 1:
#         correct +=1
#     else:
#         wrong+=1
# print('precision:', correct*1.0/(correct+wrong))
# print("Mean squared error: %.2f" % mean_squared_error(vec, target))
# print('Variance score: %.2f' % r2_score(yp, target))

