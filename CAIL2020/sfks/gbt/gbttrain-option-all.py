import json
import os

import numpy
import thulac
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import re

# dtype = object), 'params': [{'n_estimators': 20}, {'n_estimators': 30}, {'n_estimators': 40}, {'n_estimators': 50},
#                             {'n_estimators': 60}, {'n_estimators': 70},
#                             {'n_estimators': 80}], 'split0_test_score': array(
#     [0.56535078, 0.56615581, 0.56915381, 0.57136097, 0.57136097,
#      0.56360044, 0.56360044]), 'split1_test_score': array([0.58530144, 0.58559289, 0.58609119, 0.58580192, 0.58685283,
#                                                            0.58649125, 0.587132]), 'split2_test_score': array(
#     [0.59252028, 0.59277243, 0.59277789, 0.60809892, 0.60843076,
#      0.60816878, 0.60823209]), 'split3_test_score': array([0.57848586, 0.57848586, 0.57848586, 0.57848586, 0.57848586,
#                                                            0.58176138, 0.5819711]), 'split4_test_score': array(
#     [0.58598912, 0.58594433, 0.59089981, 0.59073378, 0.5933471,
#      0.59332307, 0.59498063]), 'mean_test_score': array([0.58152949, 0.58179026, 0.58348171, 0.58689629, 0.5876955,
#                                                          0.58666898, 0.58718325]), 'std_test_score': array(
#     [0.0092296, 0.00902972, 0.00870062, 0.01247042, 0.01276116,
#      0.01457938, 0.01474565]), 'rank_test_score': array([7, 6, 5, 3, 1, 4, 2], dtype=int32)}

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
param_test1 = {'n_estimators': range(130, 321, 20)}
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(100,801,200)}
param_test3 = {'min_samples_split':range(800,1900,200), 'min_samples_leaf':range(60,101,10)}
param_test4 = {'max_features':range(7,20,2)}
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}

gsearch1 = GridSearchCV(estimator = gbt, param_grid = param_test1, n_jobs=24, scoring='roc_auc',iid=False,cv=None)
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

