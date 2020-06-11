
# single = [{'TP': 15, 'FN': 1584, 'FP': 63, 'TN': 0}, {'TP': 229, 'FN': 1824, 'FP': 693, 'TN': 0}, {'TP': 1041, 'FN': 1190, 'FP': 2713, 'TN': 0}, {'TP': 910, 'FN': 1296, 'FP': 2425, 'TN': 0}]
#single = [{'TP': 11, 'FN': 1556, 'FP': 106, 'TN': 0}, {'TP': 176, 'FN': 1838, 'FP': 498, 'TN': 0}, {'TP': 1126, 'FN': 1167, 'FP': 2863, 'TN': 0}, {'TP': 890, 'FN': 1325, 'FP': 2419, 'TN': 0}]
# multi = [{'TP': 7, 'FN': 666, 'FP': 81, 'TN': 0}, {'TP': 0, 'FN': 617, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 655, 'FP': 0, 'TN': 0}, {'TP': 12, 'FN': 1074, 'FP': 152, 'TN': 0}, {'TP': 0, 'FN': 661, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 607, 'FP': 0, 'TN': 0}, {'TP': 11, 'FN': 922, 'FP': 104, 'TN': 0}, {'TP': 13, 'FN': 549, 'FP': 271, 'TN': 0}, {'TP': 0, 'FN': 786, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 758, 'FP': 0, 'TN': 0}, {'TP': 1441, 'FN': 111, 'FP': 8531, 'TN': 0}, {'TP': 0, 'FN': 309, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 407, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 513, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 504, 'FP': 0, 'TN': 0}]
multi = [{'TP': 3, 'FN': 601, 'FP': 20, 'TN': 0}, {'TP': 1, 'FN': 620, 'FP': 26, 'TN': 0}, {'TP': 0, 'FN': 629, 'FP': 9, 'TN': 0}, {'TP': 34, 'FN': 943, 'FP': 300, 'TN': 0}, {'TP': 0, 'FN': 662, 'FP': 15, 'TN': 0}, {'TP': 0, 'FN': 610, 'FP': 22, 'TN': 0}, {'TP': 31, 'FN': 923, 'FP': 159, 'TN': 0}, {'TP': 1, 'FN': 570, 'FP': 37, 'TN': 0}, {'TP': 7, 'FN': 825, 'FP': 121, 'TN': 0}, {'TP': 4, 'FN': 754, 'FP': 55, 'TN': 0}, {'TP': 1455, 'FN': 147, 'FP': 6520, 'TN': 0}]
single =[{'TP': 57, 'FN': 1619, 'FP': 218, 'TN': 0}, {'TP': 207, 'FN': 1792, 'FP': 569, 'TN': 0}, {'TP': 1264, 'FN': 1177, 'FP': 3126, 'TN': 0}, {'TP': 820, 'FN': 1476, 'FP': 2151, 'TN': 0}]
#single distribution [0.19371986648535047, 0.2489800964272469, 0.2834713808876252, 0.2738286561997775]
#multiple distribution [0.2071555643407788, 0.06868946432101206, 0.06829412927456019, 0.12205969559201423, 0.05930025696778019, 0.058904921921328325, 0.08588653884166832, 0.05159122356196877, 0.07461949001779007, 0.06424194504842855, 0.13925677011267049]


# BERT+CNN
# precision : 0.2765483990604525
# precision : 0.13969688411936365
# precision : 0.1988563488670372
# BET+CNN+GRU
# precision : 0.2791
# precision : 0.1741
# precision : 0.2253

def precision(dic):
    ttp = 0
    ttn = 0
    cnt = []
    for k in dic:
        ttp += k['TP']
        ttn += k['FP']
        cnt.append(k['TP']+ k['FN'])
    cntsum = sum(cnt)
    cnt = [i*1.0/cntsum for i in cnt]

    print('precision :', ttp*1.0 /(ttp+ttn))
    print('distribution', cnt)

def precisionall(dics):
    ttp = 0
    ttn = 0
    for dic in dics:
        for k in dic:
            ttp += k['TP']
            ttn += k['FP']

    print('precision :', ttp*1.0 /(ttp+ttn))


precision(single)
precision(multi)
precisionall([single, multi])