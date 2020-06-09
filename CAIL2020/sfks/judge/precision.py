
single = [{'TP': 20, 'FN': 1568, 'FP': 59, 'TN': 0}, {'TP': 135, 'FN': 1875, 'FP': 457, 'TN': 0}, {'TP': 1504, 'FN': 810, 'FP': 3841, 'TN': 0}, {'TP': 578, 'FN': 1599, 'FP': 1495, 'TN': 0}]
multi = [{'TP': 7, 'FN': 666, 'FP': 81, 'TN': 0}, {'TP': 0, 'FN': 617, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 655, 'FP': 0, 'TN': 0}, {'TP': 12, 'FN': 1074, 'FP': 152, 'TN': 0}, {'TP': 0, 'FN': 661, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 607, 'FP': 0, 'TN': 0}, {'TP': 11, 'FN': 922, 'FP': 104, 'TN': 0}, {'TP': 13, 'FN': 549, 'FP': 271, 'TN': 0}, {'TP': 0, 'FN': 786, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 758, 'FP': 0, 'TN': 0}, {'TP': 1441, 'FN': 111, 'FP': 8531, 'TN': 0}, {'TP': 0, 'FN': 309, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 407, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 513, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 504, 'FP': 0, 'TN': 0}]

# precision : 0.2765483990604525
# precision : 0.13969688411936365
# precision : 0.1988563488670372

def precision(dic):
    ttp = 0
    ttn = 0
    for k in dic:
        ttp += k['TP']
        ttn += k['FP']

    print('precision :', ttp*1.0 /(ttp+ttn))

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