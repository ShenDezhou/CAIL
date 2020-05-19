# CAIL2020——司法考试数据说明

## 数据说明

本数据来源于论文``JEC-QA: A Legal-Domain Question Answering Dataset``，为司法考试训练集。

训练集包含两个文件``0_train.json,1_train.json``，分别对应概念理解题和情景分析题。

两个文件均包含若干行，每行数据均为json格式，包含若干字段：

* ``answer``：代表该题的答案。
* ``id``：题目的唯一标识符。
* ``option_list``：题目每个选项的描述。
* ``statement``：题干的描述。
* ``subject``：代表该问题所属的分类，仅有部分数据含有该字段。
* ``type``：无意义字段。

实际测试数据不包含``answer``字段。

更多信息请参考https://github.com/china-ai-law-challenge/CAIL2020。