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


# {'mean_fit_time': array([ 314.39607468,  211.88898778,  201.19792786,  198.32579422,
#         320.46039462,  478.36985893,  517.79130836,  336.21576018,
#         436.97495241,  433.27550054,  617.81046972,  932.61086025,
#        1176.69754896, 1197.02157278,  896.41745391,  557.14855323,
#         652.30799012,  658.36218309,  646.01170983,  633.84223571,
#         711.02935777,  665.90113688,  634.35349836,  656.35094876]), 'std_fit_time': array([ 47.67760073,  10.35870406,   5.78110766,   4.80112004,
#          0.7795211 , 180.12838252, 159.94716485,  28.90744629,
#         21.24746597,  22.69024805, 166.17522872,   8.93793618,
#         13.69029406,  15.64577703, 182.29220986,  33.65756526,
#         13.73986338,  33.23994792,  27.90531067,  22.19200936,
#         13.54015909,  11.96153439,  11.34287522,  17.84903265]), 'mean_score_time': array([0.09016695, 0.07096419, 0.06509929, 0.07046442, 0.07007384,
#        0.195116  , 0.11571708, 0.07366934, 0.07545805, 0.07395449,
#        0.16786962, 0.21941266, 0.24241567, 0.23364697, 0.11910872,
#        0.08294702, 0.08866663, 0.08864999, 0.08760495, 0.08682637,
#        0.0947228 , 0.07848568, 0.07565536, 0.07938352]), 'std_score_time': array([0.01852831, 0.0096461 , 0.0013281 , 0.01059489, 0.00136126,
#        0.08206723, 0.06602372, 0.00547429, 0.00379918, 0.00380833,
#        0.07085279, 0.01626076, 0.00502753, 0.01508481, 0.03488571,
#        0.00279349, 0.00458982, 0.00521238, 0.00432359, 0.00456944,
#        0.00974026, 0.00081988, 0.00073851, 0.011148  ]), 'param_max_depth': masked_array(data=[3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7, 9, 9, 9, 9, 11, 11,
#                    11, 11, 13, 13, 13, 13],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_min_samples_split': masked_array(data=[100, 300, 500, 700, 100, 300, 500, 700, 100, 300, 500,
#                    700, 100, 300, 500, 700, 100, 300, 500, 700, 100, 300,
#                    500, 700],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'max_depth': 3, 'min_samples_split': 100}, {'max_depth': 3, 'min_samples_split': 300}, {'max_depth': 3, 'min_samples_split': 500}, {'max_depth': 3, 'min_samples_split': 700}, {'max_depth': 5, 'min_samples_split': 100}, {'max_depth': 5, 'min_samples_split': 300}, {'max_depth': 5, 'min_samples_split': 500}, {'max_depth': 5, 'min_samples_split': 700}, {'max_depth': 7, 'min_samples_split': 100}, {'max_depth': 7, 'min_samples_split': 300}, {'max_depth': 7, 'min_samples_split': 500}, {'max_depth': 7, 'min_samples_split': 700}, {'max_depth': 9, 'min_samples_split': 100}, {'max_depth': 9, 'min_samples_split': 300}, {'max_depth': 9, 'min_samples_split': 500}, {'max_depth': 9, 'min_samples_split': 700}, {'max_depth': 11, 'min_samples_split': 100}, {'max_depth': 11, 'min_samples_split': 300}, {'max_depth': 11, 'min_samples_split': 500}, {'max_depth': 11, 'min_samples_split': 700}, {'max_depth': 13, 'min_samples_split': 100}, {'max_depth': 13, 'min_samples_split': 300}, {'max_depth': 13, 'min_samples_split': 500}, {'max_depth': 13, 'min_samples_split': 700}], 'split0_test_score': array([0.59057455, 0.58671884, 0.58671884, 0.58898631, 0.60904886,
#        0.608076  , 0.60828148, 0.60715989, 0.61155211, 0.61079701,
#        0.6113095 , 0.61024195, 0.61881979, 0.62038347, 0.6164388 ,
#        0.61554944, 0.62577065, 0.62285206, 0.62217174, 0.62159784,
#        0.62521968, 0.62580394, 0.62461222, 0.62189093]), 'split1_test_score': array([0.58674941, 0.58634171, 0.58634171, 0.58655347, 0.59268674,
#        0.59256558, 0.59268237, 0.59112607, 0.59705003, 0.59922826,
#        0.59960321, 0.59992741, 0.60836826, 0.60578506, 0.60561341,
#        0.60405055, 0.61846094, 0.61418034, 0.61414022, 0.61276975,
#        0.61833213, 0.61494771, 0.61562094, 0.61535978]), 'split2_test_score': array([0.60492217, 0.59490154, 0.59418601, 0.59216879, 0.61739556,
#        0.61670705, 0.61665138, 0.61594241, 0.62101604, 0.62121497,
#        0.62154436, 0.61816294, 0.62541998, 0.62405661, 0.62274972,
#        0.62058896, 0.62753681, 0.62816065, 0.62715886, 0.62434014,
#        0.6306538 , 0.63309619, 0.63009355, 0.62950765]), 'split3_test_score': array([0.60733177, 0.60696967, 0.60696967, 0.59656254, 0.61685468,
#        0.61605757, 0.61682956, 0.61720585, 0.62471348, 0.62517661,
#        0.62369054, 0.6239415 , 0.62806765, 0.62733035, 0.62777682,
#        0.62649064, 0.63551574, 0.63172001, 0.63372465, 0.63161379,
#        0.63781448, 0.63381258, 0.63508292, 0.63270117]), 'split4_test_score': array([0.61010019, 0.60978151, 0.60978151, 0.61169604, 0.61859744,
#        0.61865342, 0.61758078, 0.62098629, 0.62275199, 0.62232463,
#        0.62243577, 0.62598027, 0.62642401, 0.62526126, 0.62421703,
#        0.62716568, 0.63022056, 0.62982597, 0.62787868, 0.63055808,
#        0.6318175 , 0.63233388, 0.63041854, 0.63437265]), 'mean_test_score': array([0.59993562, 0.59694265, 0.59679955, 0.59519343, 0.61091666,
#        0.61041192, 0.61040512, 0.6104841 , 0.61541673, 0.6157483 ,
#        0.61571668, 0.61565081, 0.62141994, 0.62056335, 0.61935916,
#        0.61876906, 0.62750094, 0.62534781, 0.62501483, 0.62417592,
#        0.62876752, 0.62799886, 0.62716563, 0.62676644]), 'std_test_score': array([0.00942754, 0.00986357, 0.00989728, 0.00890546, 0.00971647,
#        0.00962676, 0.00949089, 0.01068695, 0.01023581, 0.00958595,
#        0.00918431, 0.00957314, 0.00724348, 0.00772652, 0.00779007,
#        0.00849045, 0.00559196, 0.00631684, 0.00655743, 0.0068239 ,
#        0.0065756 , 0.00712355, 0.00665789, 0.00713403]), 'rank_test_score': array([21, 22, 23, 24, 17, 19, 20, 18, 16, 13, 14, 15,  9, 10, 11, 12,  3,
#         6,  7,  8,  1,  2,  4,  5], dtype=int32)}

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

