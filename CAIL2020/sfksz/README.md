项目使用方法请参考[这里](https://github.com/haoxizhong/pytorch-worker)。

该模型为用Attention整合题干和选项，然后进行预测的模型。

数据预处理命令： ``python utils\cutter.py --data input --output data/cutted --gen_word2id``
训练命令： ``python3 train.py --config config/model.config --gpu 0``。


#2020.07.15   hidden 256, number of layer 2, epoch 10, TEST F1: 17.21
#2020.07.15   lstm encoder tripled hidden neurons and add one more layer from 2 to 3, batch size lowered to 32, epoch lowered to 10.   TEST F1: 24.9
#2020.07.16   lstm encoder tripled hidden neurons and add one more layer from 2 to 3, batch size lowered to 32, epoch lowered to 14.   TEST F1: 27.04
#2020.07.16   lstm encoder tripled hidden neurons and add one more layer from 2 to 3, batch size lowered to 32, epoch lowered to 19.   TEST F1: 26.41


1      train  2635/2634  20:06/ 0:00    5.223   {"micro_precision": 0.038}
1      valid  25/25       0:02/ 0:00    2.497   {"micro_precision": 0.06}
2      train  2635/2634  20:38/ 0:00    5.060   {"micro_precision": 0.049}
2      valid  25/25       0:02/ 0:00    2.351   {"micro_precision": 0.1}
3      train  2635/2634  20:32/ 0:00    4.693   {"micro_precision": 0.111}
3      valid  25/25       0:02/ 0:00    1.861   {"micro_precision": 0.21}
4      train  2635/2634  20:35/ 0:00    3.619   {"micro_precision": 0.329}
4      valid  25/25       0:02/ 0:00    1.020   {"micro_precision": 0.69}
5      train  2635/2634  20:35/ 0:00    2.450   {"micro_precision": 0.565}
5      valid  25/25       0:02/ 0:00    0.494   {"micro_precision": 0.88}
6      train  2635/2634  20:32/ 0:00    1.606   {"micro_precision": 0.738}
6      valid  25/25       0:02/ 0:00    0.273   {"micro_precision": 0.95}
7      train  2635/2634  20:33/ 0:00    1.094   {"micro_precision": 0.834}
7      valid  25/25       0:02/ 0:00    0.163   {"micro_precision": 0.98}
8      train  2635/2634  20:31/ 0:00    0.793   {"micro_precision": 0.889}
8      valid  25/25       0:02/ 0:00    0.112   {"micro_precision": 0.98}
9      train  2635/2634  20:34/ 0:00    0.614   {"micro_precision": 0.919}
9      valid  25/25       0:02/ 0:00    0.089   {"micro_precision": 0.98}