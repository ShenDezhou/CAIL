import json
import os
import jieba

from PythonROUGE import PythonROUGE

input_path = "../sfzyza/data/test.json"
pre_input_path = "../sfzyza/data/pretest.json"
output_path = "../sfzyza/data/result.json"
phase1_file = "../sfzyza/data/para_content_test1.csv"


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

if __name__ == "__main__":
    os.chdir('sfzyy')
    os.system(
        "python main.py -in_file={} -out_file={}".format(input_path, pre_input_path))
    os.chdir('../sfzyzb')
    os.system(
        "python main.py -in_file={} -out_file={}".format(pre_input_path, phase1_file))
    os.chdir('../sfzyza')
    os.system(
        "python main.py -in_file={} -temp_file={} -out_file={}".format(pre_input_path, phase1_file, output_path))

    os.chdir('..')
    get_path("prediction")
    pred_list = []
    with open(output_path.replace("../",""),'r', encoding='utf8') as fr:
        for line in fr:
            item = json.loads(line)
            pred_list.append(os.path.join("prediction", item['id'] + ".txt"))
            with open(os.path.join("prediction", item['id']+".txt"),'w', encoding='utf8') as fw:
                fw.write(item['summary'])

    get_path("gold")
    gold_list = []
    with open(input_path.replace("../",""),'r', encoding='utf8') as fr:
        for line in fr:
            item = json.loads(line)
            gold_list.append([os.path.join("gold", item['id'] + ".txt")])
            with open(os.path.join("gold", item['id']+".txt"),'w', encoding='utf8') as fw:
                fw.write(item['summary'])

    recall_list,precision_list,F_measure_list = PythonROUGE(pred_list,gold_list)
    print(precision_list, recall_list, F_measure_list)
    print('weighted F1-score:', 0.2*F_measure_list[0] + 0.4*F_measure_list[1]+ 0.4*F_measure_list[2])


