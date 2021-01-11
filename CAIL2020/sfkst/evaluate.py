"""Evaluate model and calculate results for SMP-CAIL2020-Argmine.

Author: Tsinghuaboy tsinghua9boy@sina.com
"""



def precision(dic):
    ttp = 0
    ttn = 0
    cnt = []
    for k in dic:
        ttp += k['TP']
        ttn += k['FP']
        # cnt.append(k['TP']+ k['FP'])
        if ttp+ttn==0:
            cnt.append(0)
        else:
            cnt.append(ttp * 1.0 / (ttp + ttn))
    # cntsum = sum(cnt)
    # cnt = [i*1.0/cntsum for i in cnt]
    return cnt
    # print('precision :', ttp*1.0 /(ttp+ttn))
    # print('distribution', cnt)


def recall(dic):
    ttp = 0
    ttn = 0
    cnt = []
    for k in dic:
        ttp += k['TP']
        ttn += k['FN']
        # cnt.append(k['TP'] + k['FN'])
        if ttp+ttn==0:
            cnt.append(0)
        else:
            cnt.append(ttp * 1.0 / (ttp + ttn))
    # cntsum = sum(cnt)
    # cnt = [i * 1.0 / cntsum for i in cnt]
    return cnt
    # print('precision :', ttp * 1.0 / (ttp + ttn))
    # print('distribution', cnt)



def p_f1(dic):
    pl = precision(dic)
    rl = recall(dic)
    fl = []
    for i in range(len(dic)):
        p = pl[i]
        r = rl[i]
        if p==0 or r==0:
            f=0
        else:
            f = 2.0/((1.0/p) + (1.0/r))
        fl.append(f)
    tpl = sum(pl)
    tfl = sum(fl)
    return tpl/len(pl), tfl/len(fl)
