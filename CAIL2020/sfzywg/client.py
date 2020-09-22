#!/usr/local/bin/python
# -*- coding: utf-8 -*-

#[0.73569, 0.73569, 0.73569] [0.73569, 0.73569, 0.73569] [0.73569, 0.73569, 0.73569]
import json
import os
import jieba
import requests
import re

from PythonROUGE import PythonROUGE

# input_path = "data/sfzy_small.json"    #4047
input_path = "data/sfzy_large.json"  #9484
# pre_input_path = "../sfzyza/data/pretest.json"
#output_path = "data/test.json"
# phase1_file = "../sfzyza/data/para_content_test1.csv"


def get_path(path):
    """Create the path if it does not exist.

    Args:
        path: path to be used

    Returns:
        Existed path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


pattern = '<string.*>([^<]*)</string>'


def getSummary(text):
    request = {'text': text}
    r = requests.post(
        'http://casm.pkulaw.cn/lawqa/wbser_keys.asmx/wbser_getKeySentence2',
        data=request,
        timeout=12.5)
    if r.status_code == 200:
        body = re.findall(pattern, r.text)
        resp = json.loads(body[0])
        resp = resp["d2"].replace('\n',"", 10**10).replace(r'\n',"", 10**10)
        print(resp)
        return resp
    return ""

#getSummary(r"徐绍勤、徐艳华等与徐少清继承纠纷一审民事判决书\n辽宁省北镇市人民法院\n民 事 判 决 书\n（2017）辽0782民初2218号\n原告：徐绍勤，男，1951年7月12日生，满族，农民，住辽宁省北镇市。\n委托诉讼代理人：张凤林，系辽宁名崛律师事务所律师。\n原告:徐艳华,女,1959年3月8日生,满族，农民，住辽宁省北镇市。\n原告:徐艳秋,女,1967年8月8日生,满族，农民，住辽宁省北镇市。\n被告：徐少清，男，1956年4月4日生，满族，职工，住辽宁省北镇市。\n委托诉讼代理人：徐超(被告之子),住辽宁省北镇市北镇街道办事处正兰旗胡同19号。\n委托诉讼代理人：钱常惠，系北镇市北镇法律服务所法律工作者。\n原告徐绍勤、徐艳华、徐艳秋与被告徐少清继承纠纷一案，本院于2017年9月14日立案受理后，依法适用简易程序，公开开庭进行了审理。\n原、被告及委托诉讼代理人均到庭参加了诉讼。\n本案现已审理终结。\n原告徐绍勤向本院提出诉讼请求：1.请求继承父、母遗产房屋两间、存款4万元及利息，分割父亲的丧葬费、丧葬补助金137007元；\n2.要求被告承担诉讼费。\n事实和理由：原、被告均系徐世国、朱凌云的子女。\n父徐世国（北镇市城建局退休）、母朱凌云在北镇市万紫山社区两间民房居住。\n母亲2014年11月去世（未对母亲遗产进行继承），父亲由原、被告轮流照顾生活起居，2017年4月去世。\n父、母安葬费用均用父、母积蓄。\n父、母留下的两间平房（现已动迁，被告得一户住宅楼，富源城堡小区），银行存款4万元，均由被告占有。\n父亲去世的丧葬费、丧葬补助金137007元也由被告支取。\n原告徐艳华、徐艳秋（依法追加）称，父亲的遗嘱是真实的，听从法院判决。\n被告徐少清辩称，答辩人的母亲于2013年11月去世。\n答辩人母亲去世后，父亲由答辩人一人照顾生活起居至父亲去世，父亲去世时间是2017年5月。\n答辩人的母亲去世时，答辩人的父亲就将母亲的遗产（包括两间房屋的一半）进行了分割，给原、被告四人每人5000元。\n关于两间平房，现已动迁，答辩人的父亲遗嘱将两间平房给了答辩人。\n存款4万元及利息780元和丧抚费137007元确经答辩人手取出，但这些钱答辩人父亲在世的时候，因为答辩人一人独自照顾父亲的生活起居，答辩人的父亲亲口交待，除去用于丧葬之外剩余的钱,均归答辩人所有。\n本院经审理认定事实如下：原、被告均系被继承人徐世国、朱凌云的子女。\n二被继承人原在北镇市万紫山社区有两间民房，现已动迁，动迁后置换房屋还没有进行建设。\n被继承人朱凌云于2013年11月去世，当时原、被告父亲(被继承人徐世国)分给原、被告四人每人人民币5000元。\n其后被继承人徐世国主要由被告照顾生活起居，被继承人徐世国于2017年5月去世，被继承人徐世国去世后，由被告进行了发丧。\n被继承人徐世国生前留下自书遗嘱：”房子如有动迁，房子享楼由我二儿子由徐绍青继承。\n父徐世国。\n2016.1.24”。\n被继承人徐世国去世后，其银行存款4万元及利息780元和徐世国的丧葬费、丧葬补助金137007元由被告支取。\n本院认为，原、被告均系二被继承人的子女，故依法对被继承人的遗产均享有继承权。\n关于被继承人徐世国生前留下的自书遗嘱，原告徐艳华、徐艳秋对其真实性予以确认，原告徐绍勤虽对该遗嘱真实性持否定态度，但未在本院限定时间内预交鉴定费用，故本院对该份遗嘱的真实性予以确认。\n在被继承人朱凌云去世时，被继承人徐世国虽分给原、被告四人每人人民币5000元，但不能说明该笔钱是对被继承人朱凌云遗产（两间房屋的一半）的处分。\n故被继承人徐世国留下的自书遗嘱虽是真实的，但其处分被继承人朱凌云遗产（两间房屋的一半）是无效的。\n被继承人朱凌云遗产（两间房屋的一半）应按法定继承处理。\n现该房屋已动迁，置换房屋还未建设，暂无法确认具体继承数额，可待置换房屋建成后另案处理。\n关于4万元存款及780元利息，被告没有证据证明被继承人生前答应由其继承该款，故应按法定继承处理。\n关于137007元丧抚费，不是被继承人的遗产。\n被继承人徐世国去世后，被告对其进行了发丧，故其中的丧葬费28574元，应归被告所有。\n其余108433元，系抚恤金。\n可参照遗产继承进行分割。\n由于被告对被继承人徐世国进了主要的赡养义务，故在遗产继承及抚恤金分割时应多分。\n综上，依照《中华人民共和国继承法》第三条、第五条、第十条一款、第十三条三款及《中华人民共和国民事诉讼法》第一百四十二条之规定，判决如下：\n一、由三原告各继承被继承人徐世国存款及利息40780元的15%，即6117元，由被告继承55%，即22429元；\n二、徐世国的抚恤金108433元，由三原告各分得15%，即16265元，由被告分得55%，即59638元；\n三、徐世国的丧葬费28574元归被告所有；\n以上一、二项三原告应得数额，由被告于本判决生效后5日内给付。\n如果未按本判决指定的期限履行金钱给付义务的，应当依照《中华人民共和国民事诉讼法》第二百五十三条的规定，加倍支付迟延履行期间的债务利息。\n案件受理费4300元，减半收取2150元，由三原告各负担322元,由被告徐少清负担1184元。\n如不服本判决，可在判决书送达之日起15日内，向本院递交上诉状，并按对方当事人的人数提出副本，上诉于辽宁省锦州市中级人民法院。\n审判员　　侯百吉\n二〇一七年十一月九日\n书记员　　董　健")

if __name__ == "__main__":
    # os.chdir('sfzyy')
    # os.system(
    #     "python main.py -in_file={} -out_file={}".format(input_path, pre_input_path))
    # os.chdir('../sfzyzb')
    # os.system(
    #     "python main.py -in_file={} -out_file={}".format(pre_input_path, phase1_file))
    # os.chdir('../sfzyza')
    # os.system(
    #     "python main.py -in_file={} -temp_file={} -out_file={}".format(pre_input_path, phase1_file, output_path))
    #
    # os.chdir('..')
    get_path("prediction")
    pred_list = []
    with open(input_path.replace("../",""),'r', encoding='utf8') as fr:
        for line in fr:
            item = json.loads(line)
            pred_list.append(os.path.join("prediction", item['id'] + ".txt"))
            with open(os.path.join("prediction", item['id']+".txt"),'w', encoding='utf8') as fw:
                sentences = [sub['sentence'] for sub in item['text']]
                whole = "".join(sentences)
                fw.write(getSummary(whole))
    print(pred_list)

    get_path("gold")
    gold_list = []
    with open(input_path.replace("../",""),'r', encoding='utf8') as fr:
        for line in fr:
            item = json.loads(line)
            gold_list.append([os.path.join("gold", item['id'] + ".txt")])
            with open(os.path.join("gold", item['id']+".txt"),'w', encoding='utf8') as fw:
                fw.write(item['summary'])
    print(gold_list)

    recall_list,precision_list,F_measure_list = PythonROUGE(pred_list,gold_list)
    print(precision_list, recall_list, F_measure_list)
    print('weighted F1-score:', 0.2*F_measure_list[0] + 0.4*F_measure_list[1]+ 0.4*F_measure_list[2])


