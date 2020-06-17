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

# {'mean_fit_time': array([431.42769513, 313.74814315, 308.92651234, 308.17050109,
#        566.40047712, 417.28646412, 331.63456535, 323.77848864,
#        316.20123758, 335.68334136, 554.3873744 , 684.18733182,
#        665.75850749, 670.51941614, 685.54875078, 677.35225258,
#        518.13547063, 336.52881284, 319.05661769, 313.14093561,
#        315.3283844 , 319.05867419, 311.85861039, 314.37540641,
#        316.19316535, 312.84743505, 309.23655725, 312.70924892,
#        307.38148122, 304.51115799]), 'std_fit_time': array([ 57.57890849,   7.67538528,   2.12093362,   3.21810247,
#         30.9627074 ,  81.39351786,  13.15680577,  13.02504226,
#         13.95069054,  35.38607002, 116.97619128,   9.1881518 ,
#         17.57829546,  14.41317132,   5.89755463,   7.86861274,
#         68.72970845,   6.77551098,   5.62368721,   4.02537727,
#          7.85329986,  10.07367154,  14.15082213,  10.95045514,
#         12.51403738,  11.24451999,  11.79909996,   9.55087717,
#          3.00416662,   2.58857202]), 'mean_score_time': array([0.08066554, 0.06881123, 0.06873121, 0.06841698, 0.16651769,
#        0.0749104 , 0.07492037, 0.0736022 , 0.07127275, 0.10994749,
#        0.18216329, 0.19886622, 0.13775125, 0.19447861, 0.19960017,
#        0.18823633, 0.12451777, 0.07252727, 0.07013226, 0.07006378,
#        0.07087927, 0.07759113, 0.07016616, 0.07108736, 0.07086368,
#        0.06907477, 0.06914573, 0.0698143 , 0.06973915, 0.06712914]), 'std_score_time': array([0.01272009, 0.00071331, 0.00084774, 0.00037664, 0.05256967,
#        0.00417877, 0.00398051, 0.00227171, 0.00301326, 0.07339607,
#        0.01139775, 0.02621701, 0.0523646 , 0.02323532, 0.01942443,
#        0.01966959, 0.02811216, 0.00219947, 0.00067295, 0.00282538,
#        0.00263243, 0.01586017, 0.0044391 , 0.00459553, 0.00386831,
#        0.00250871, 0.00411311, 0.00399557, 0.00141285, 0.001685  ]), 'param_min_samples_leaf': masked_array(data=[60, 60, 60, 60, 60, 60, 70, 70, 70, 70, 70, 70, 80, 80,
#                    80, 80, 80, 80, 90, 90, 90, 90, 90, 90, 100, 100, 100,
#                    100, 100, 100],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'param_min_samples_split': masked_array(data=[800, 1000, 1200, 1400, 1600, 1800, 800, 1000, 1200,
#                    1400, 1600, 1800, 800, 1000, 1200, 1400, 1600, 1800,
#                    800, 1000, 1200, 1400, 1600, 1800, 800, 1000, 1200,
#                    1400, 1600, 1800],
#              mask=[False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False, False, False,
#                    False, False, False, False, False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'min_samples_leaf': 60, 'min_samples_split': 800}, {'min_samples_leaf': 60, 'min_samples_split': 1000}, {'min_samples_leaf': 60, 'min_samples_split': 1200}, {'min_samples_leaf': 60, 'min_samples_split': 1400}, {'min_samples_leaf': 60, 'min_samples_split': 1600}, {'min_samples_leaf': 60, 'min_samples_split': 1800}, {'min_samples_leaf': 70, 'min_samples_split': 800}, {'min_samples_leaf': 70, 'min_samples_split': 1000}, {'min_samples_leaf': 70, 'min_samples_split': 1200}, {'min_samples_leaf': 70, 'min_samples_split': 1400}, {'min_samples_leaf': 70, 'min_samples_split': 1600}, {'min_samples_leaf': 70, 'min_samples_split': 1800}, {'min_samples_leaf': 80, 'min_samples_split': 800}, {'min_samples_leaf': 80, 'min_samples_split': 1000}, {'min_samples_leaf': 80, 'min_samples_split': 1200}, {'min_samples_leaf': 80, 'min_samples_split': 1400}, {'min_samples_leaf': 80, 'min_samples_split': 1600}, {'min_samples_leaf': 80, 'min_samples_split': 1800}, {'min_samples_leaf': 90, 'min_samples_split': 800}, {'min_samples_leaf': 90, 'min_samples_split': 1000}, {'min_samples_leaf': 90, 'min_samples_split': 1200}, {'min_samples_leaf': 90, 'min_samples_split': 1400}, {'min_samples_leaf': 90, 'min_samples_split': 1600}, {'min_samples_leaf': 90, 'min_samples_split': 1800}, {'min_samples_leaf': 100, 'min_samples_split': 800}, {'min_samples_leaf': 100, 'min_samples_split': 1000}, {'min_samples_leaf': 100, 'min_samples_split': 1200}, {'min_samples_leaf': 100, 'min_samples_split': 1400}, {'min_samples_leaf': 100, 'min_samples_split': 1600}, {'min_samples_leaf': 100, 'min_samples_split': 1800}], 'split0_test_score': array([0.60240528, 0.60430489, 0.60460807, 0.60423939, 0.60423148,
#        0.60423148, 0.60171431, 0.60318548, 0.60396459, 0.60426013,
#        0.60426013, 0.60426013, 0.60340625, 0.60200303, 0.60200303,
#        0.60050758, 0.60050758, 0.60050758, 0.60356262, 0.60343354,
#        0.60249915, 0.60167529, 0.60167529, 0.60167529, 0.60598645,
#        0.60644218, 0.60603885, 0.60624188, 0.60624188, 0.60624188]), 'split1_test_score': array([0.59046621, 0.59045529, 0.59295008, 0.59295008, 0.59295008,
#        0.59295008, 0.58982027, 0.58977934, 0.59077949, 0.59077949,
#        0.59077949, 0.59089493, 0.58949362, 0.58977743, 0.58994116,
#        0.58994116, 0.58994116, 0.58994116, 0.5902318 , 0.59041654,
#        0.59045693, 0.59045693, 0.59045693, 0.59045693, 0.58889871,
#        0.59147428, 0.59006997, 0.59006997, 0.59006997, 0.59006997]), 'split2_test_score': array([0.61297824, 0.61254435, 0.61254435, 0.61254435, 0.61254435,
#        0.61254435, 0.61310214, 0.61439128, 0.61439128, 0.61439128,
#        0.61439128, 0.61439128, 0.61311169, 0.61231402, 0.61231402,
#        0.61231402, 0.61231402, 0.61231402, 0.61320529, 0.61322603,
#        0.61322603, 0.61322603, 0.61322603, 0.61322603, 0.61547549,
#        0.61478944, 0.61478944, 0.61478944, 0.61478944, 0.61478944]), 'split3_test_score': array([0.61382192, 0.61354229, 0.61354229, 0.61354229, 0.61354229,
#        0.61354229, 0.6153058 , 0.61410646, 0.61410646, 0.61410646,
#        0.61410646, 0.61410646, 0.61553   , 0.61439155, 0.61439155,
#        0.61439155, 0.61439155, 0.61439155, 0.61290412, 0.61427058,
#        0.61427058, 0.61427058, 0.61427058, 0.61427058, 0.61503764,
#        0.61472388, 0.61472388, 0.61472388, 0.61472388, 0.61472388]), 'split4_test_score': array([0.62588169, 0.62588169, 0.62588169, 0.62600457, 0.62551823,
#        0.62551823, 0.62616978, 0.62616978, 0.62616978, 0.62623068,
#        0.62622112, 0.62622112, 0.6249658 , 0.6249658 , 0.6249658 ,
#        0.62482516, 0.62431561, 0.62431561, 0.62409005, 0.62409005,
#        0.62409005, 0.62381534, 0.62381534, 0.62381534, 0.62460234,
#        0.62460234, 0.62460234, 0.62406083, 0.62406083, 0.62406083]), 'mean_test_score': array([0.60911067, 0.6093457 , 0.6099053 , 0.60985614, 0.60975728,
#        0.60975728, 0.60922246, 0.60952647, 0.60988232, 0.60995361,
#        0.6099517 , 0.60997478, 0.60930147, 0.60869037, 0.60872311,
#        0.6083959 , 0.60829399, 0.60829399, 0.60879877, 0.60908735,
#        0.60890855, 0.60868883, 0.60868883, 0.60868883, 0.61000013,
#        0.61040642, 0.61004489, 0.6099772 , 0.6099772 , 0.6099772 ]), 'std_test_score': array([0.01192476, 0.01169221, 0.01087177, 0.01094485, 0.01080294,
#        0.01080294, 0.01242649, 0.01226214, 0.0118608 , 0.01184858,
#        0.01184596, 0.01180862, 0.01204814, 0.01194105, 0.01188925,
#        0.01203348, 0.01189527, 0.01189527, 0.01133353, 0.0113986 ,
#        0.01148401, 0.01150778, 0.01150778, 0.01150778, 0.01208271,
#        0.01107715, 0.01158765, 0.0114393 , 0.0114393 , 0.0114393 ]), 'rank_test_score': array([19, 16, 10, 12, 13, 13, 18, 15, 11,  8,  9,  7, 17, 24, 23, 28, 29,
#        29, 22, 20, 21, 25, 25, 25,  3,  1,  2,  4,  4,  4], dtype=int32)}

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

gsearch1 = GridSearchCV(estimator = gbt, param_grid = param_test3, n_jobs=24, scoring='roc_auc',iid=False,cv=None)
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

