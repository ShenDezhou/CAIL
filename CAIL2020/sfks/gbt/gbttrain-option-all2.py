import json
import os

import numpy
import thulac
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import re

#training score : 0.563  0.554 0.611
#319/7175  4.4%
#339/7459
from sklearn.model_selection import GridSearchCV


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


data_path="../input"
filename_list="0_train.json,1_train.json".replace(" ", "").split(",")

# spattern = '[何|哪|那][一|个|项|种|者|罪]|\(\)|谁'
# mpattern = '[何|哪|那][些]|[几][个|项|种|者|类]'
# spattern = '[何|哪|那][一|个|项|种|者]|\(\)|如何|何罪|什么|谁|怎[么|样]'
# mpattern = '[何|哪|那][些]|[几][个|项|种|者]'

spattern = '下列.*[何|哪|那](一|个).*'
mpattern = '下列.*[何|哪|那](几|些).*'
xpattern = '下列(.*)'

# pattern = '[何|哪|那|几][一|个|项|些|种|者]|\(\)|如何|何罪|什么|谁|怎[么|样]'

spr = re.compile(spattern)
mpr = re.compile(mpattern)
xpr = re.compile(xpattern)
statement =[]
target=[]
ignoren = 0
for filename in filename_list:
    f = open(os.path.join(data_path, filename), "r", encoding='utf-8')
    for line in f:
        data = json.loads(line)
        data["answer"] = [i for i in data["answer"] if i !='。']
        if len(data["answer"]) == 0:
            ignoren += 1
            print('*')
            continue
        if (spr.search(data["statement"]) and len(data["answer"]) == 1) or (mpr.search(data["statement"]) and len(data["answer"]) > 1):
            ignoren +=1
            #print('.')
            continue

        if (spr.search(data["statement"]) and len(data["answer"]) != 1) or (mpr.search(data["statement"]) and len(data["answer"]) == 1):
            print('X', data["statement"], data["answer"])

        temp = data["statement"]
        #for op in data["option_list"].values():
        #    temp += "。"
        #    temp += op
        statement.append(temp)

        if len(data["answer"]) == 1:
            target.append(0)
        else:
            target.append(1)
print('ignore n:', ignoren)


# vec=None
vec = numpy.load("statement_vec.npz")["arr_0"]

if vec is None:
    train_data = cut_text(statement)
    print('train tfidf...', print_mem())
    tfidf = train_tfidf(train_data)
    joblib.dump(tfidf,'statement_tfidf.model', compress=3)
    vec = tfidf.transform(train_data)
    numpy.savez_compressed("statement_vec.npz", vec.todense())

print('train gbt...', print_mem())
gbt = GradientBoostingClassifier(learning_rate = 0.01,
                                 n_estimators = 100,
                                 max_depth = 5,
                                 min_samples_leaf = 20,
                                 min_samples_split = 120,
                                 # max_features=9,
                                 # subsample=0.7,
                                 verbose=1,
                                 )
param_test1 = {'n_estimators': range(20, 81, 10)}
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}

gsearch1 = GridSearchCV(estimator = gbt, param_grid = param_test2, n_jobs=24, scoring='roc_auc',iid=False,cv=None)
gsearch1.fit(vec, target)

print(gsearch1.cv_results_)
joblib.dump(gbt, 'statement_som_gbt.model')

yp = gbt.predict(vec)
print("training score : %.3f " % gbt.score(vec, target))
correct=0
wrong=0
for i in range(len(yp)):
    if target[i]==0 and yp[i]==0:
        correct += 1
    elif target[i] == 1 and yp[i] == 1:
        correct +=1
    else:
        wrong+=1
print('precision:', correct*1.0/(correct+wrong))
# print("Mean squared error: %.2f" % mean_squared_error(vec, target))
# print('Variance score: %.2f' % r2_score(yp, target))

