# CAIL2020——司法摘要数据说明

## 数据说明

文件中每一行代表一个样本，包含若干字段：

* ``id``：样本唯一标识符。
* ``summary``：样本的摘要内容。
* ``text``：将原裁判文书按照句子切分开，内含2个字段。
    * ``sentence``：表示句子的内容。
    * ``label``：表示句子的重要度。

实际测试数据不包含``summary``字段和``text``中的``label``。

更多信息请参考https://github.com/china-ai-law-challenge/CAIL2020。