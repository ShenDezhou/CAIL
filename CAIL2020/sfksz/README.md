项目使用方法请参考[这里](https://github.com/haoxizhong/pytorch-worker)。

该模型为用Attention整合题干和选项，然后进行预测的模型。

数据预处理命令： ``python utils\cutter.py --data input --output data/cutted --gen_word2id``
训练命令： ``python3 train.py --config config/model.config --gpu 0``。


#2020.07.15   TEST F1: 17.21
#2020.07.15   lstm encoder tripled hidden neurons and add one more layer from 2 to 3, batch size lowered to 32, epoch lowered to 10.   TEST F1: