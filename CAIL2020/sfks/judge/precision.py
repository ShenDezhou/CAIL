
single = [{'TP': 15, 'FN': 1584, 'FP': 63, 'TN': 0}, {'TP': 229, 'FN': 1824, 'FP': 693, 'TN': 0}, {'TP': 1041, 'FN': 1190, 'FP': 2713, 'TN': 0}, {'TP': 910, 'FN': 1296, 'FP': 2425, 'TN': 0}]
# multi = [{'TP': 7, 'FN': 666, 'FP': 81, 'TN': 0}, {'TP': 0, 'FN': 617, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 655, 'FP': 0, 'TN': 0}, {'TP': 12, 'FN': 1074, 'FP': 152, 'TN': 0}, {'TP': 0, 'FN': 661, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 607, 'FP': 0, 'TN': 0}, {'TP': 11, 'FN': 922, 'FP': 104, 'TN': 0}, {'TP': 13, 'FN': 549, 'FP': 271, 'TN': 0}, {'TP': 0, 'FN': 786, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 758, 'FP': 0, 'TN': 0}, {'TP': 1441, 'FN': 111, 'FP': 8531, 'TN': 0}, {'TP': 0, 'FN': 309, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 407, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 513, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 504, 'FP': 0, 'TN': 0}]
multi = [{'TP': 2322, 'FN': 17, 'FP': 8187, 'TN': 0}, {'TP': 0, 'FN': 609, 'FP': 1, 'TN': 0}, {'TP': 0, 'FN': 662, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 1067, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 587, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 658, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 967, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 536, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 805, 'FP': 0, 'TN': 0}, {'TP': 0, 'FN': 793, 'FP': 4, 'TN': 0}, {'TP': 12, 'FN': 1542, 'FP': 51, 'TN': 0}]

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