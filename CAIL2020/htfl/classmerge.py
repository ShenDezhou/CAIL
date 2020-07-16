from difflib import SequenceMatcher
class_dic = ['买卖合同', '承包合同', '技术合同', '知识产权合同', '其他类合同', '房地产合同', '代理合同', '居间合同', '转让合同', '贸易合同', '劳动合同', '托管合同',
                 '仓储合同', '服务合同', '经营合同', '储运合同', '租赁合同', '其他合同', '运输合同', '承揽合同', '借款合同', '供用合同', '金融合同', '房产合同']

def merge(class_dic):
    cl = []
    for cls in class_dic:
        if len(cl)==0:
            cl.append(cls)
        else:
            meters = [SequenceMatcher(None, cls, c).ratio() for c in cl]
            final_meter = max(meters)
            if final_meter <= 0.8:
                cl.append(cls)
    return cl

#22 classes
classx_dic = ['买卖合同', '承包合同', '技术合同', '知识产权合同', '其他类合同', '房地产合同', '代理合同', '居间合同', '转让合同', '贸易合同', '劳动合同', '托管合同', '仓储合同', '服务合同', '经营合同', '储运合同', '租赁合同', '运输合同', '承揽合同', '借款合同', '供用合同', '金融合同']

def indic(cls, class_dic=classx_dic):
    if cls in class_dic:
        return class_dic.index(cls)

    meters = [SequenceMatcher(None, cls, c).ratio() for c in class_dic]
    return max(range(len(meters)), key=meters.__getitem__)


def match(gold, pred):
    return SequenceMatcher(None, gold, pred).ratio() > 0.8

if __name__ =="__main__":
    # merged = merge(class_dic)
    # print(len(merged), merged)
    print(indic('房地产合同', classx_dic))
