#Introduction
从网上获取的合同模板，根据网上标注的分类，将合同分为24类。一部分数据没有分类，作为测试数据集。
使用清洗过的308.5MB，59266行标注数据作为训练集；将2400行数据作为验证集，验证集参与训练，24类每类100条。

训练过程如图1所示： 

![ALT](science/htfl-train.svg)


#BERT ACC:79.73%
#CNN  ACC:32.67%


#last 2 layer of hidden states.ACC=92.06%

[xla:0](100) Loss=0.55811 Rate=47.74 GlobalRate=3.36 Time=Mon Aug 24 08:16:56 2020
[xla:0](200) Loss=0.52764 Rate=136.41 GlobalRate=6.57 Time=Mon Aug 24 08:17:28 2020
[xla:0](300) Loss=0.51012 Rate=171.87 GlobalRate=9.67 Time=Mon Aug 24 08:18:01 2020
[xla:0](400) Loss=0.56252 Rate=186.87 GlobalRate=12.68 Time=Mon Aug 24 08:18:34 2020
[xla:0](500) Loss=0.49166 Rate=192.49 GlobalRate=15.59 Time=Mon Aug 24 08:19:06 2020
[xla:0](600) Loss=0.57549 Rate=194.91 GlobalRate=18.41 Time=Mon Aug 24 08:19:39 2020
[xla:0](700) Loss=0.49880 Rate=195.25 GlobalRate=21.15 Time=Mon Aug 24 08:20:12 2020
[xla:0](800) Loss=0.57504 Rate=195.36 GlobalRate=23.79 Time=Mon Aug 24 08:20:44 2020
[xla:0](900) Loss=0.52771 Rate=193.73 GlobalRate=26.36 Time=Mon Aug 24 08:21:18 2020
[xla:0](1000) Loss=0.49757 Rate=194.70 GlobalRate=28.85 Time=Mon Aug 24 08:21:50 2020
[xla:0](1100) Loss=0.52536 Rate=193.57 GlobalRate=31.27 Time=Mon Aug 24 08:22:23 2020
[xla:0](1200) Loss=0.51927 Rate=194.55 GlobalRate=33.62 Time=Mon Aug 24 08:22:56 2020
[xla:0](1300) Loss=0.58093 Rate=195.48 GlobalRate=35.90 Time=Mon Aug 24 08:23:29 2020
[xla:0](1400) Loss=0.54839 Rate=194.40 GlobalRate=38.12 Time=Mon Aug 24 08:24:02 2020
[xla:0](1500) Loss=0.49317 Rate=193.83 GlobalRate=40.28 Time=Mon Aug 24 08:24:35 2020
[xla:0](1600) Loss=0.50309 Rate=194.68 GlobalRate=42.38 Time=Mon Aug 24 08:25:08 2020
[xla:0](1700) Loss=0.53198 Rate=195.11 GlobalRate=44.42 Time=Mon Aug 24 08:25:41 2020
[xla:0](1800) Loss=0.43453 Rate=194.74 GlobalRate=46.41 Time=Mon Aug 24 08:26:13 2020
[xla:0](1900) Loss=0.58225 Rate=193.82 GlobalRate=48.34 Time=Mon Aug 24 08:26:47 2020
[xla:0](2000) Loss=0.51234 Rate=194.43 GlobalRate=50.23 Time=Mon Aug 24 08:27:19 2020
[xla:0](2100) Loss=0.45219 Rate=195.17 GlobalRate=52.07 Time=Mon Aug 24 08:27:52 2020
[xla:0](2200) Loss=0.56844 Rate=196.15 GlobalRate=53.87 Time=Mon Aug 24 08:28:25 2020
[xla:0](2300) Loss=0.55873 Rate=194.26 GlobalRate=55.61 Time=Mon Aug 24 08:28:58 2020
[xla:0](2400) Loss=0.51381 Rate=193.35 GlobalRate=57.31 Time=Mon Aug 24 08:29:31 2020
[xla:0](2500) Loss=0.53790 Rate=192.59 GlobalRate=58.97 Time=Mon Aug 24 08:30:04 2020
Finished training epoch 0
[xla:0](0) Acc=0.84375 Rate=0.00 GlobalRate=0.00 Time=Mon Aug 24 08:30:38 2020
[xla:0] Accuracy=88.10%
Finished test epoch 0, valid=88.10
[xla:0](0) Loss=0.51087 Rate=108.98 GlobalRate=108.98 Time=Mon Aug 24 08:30:47 2020
[xla:0](100) Loss=0.51433 Rate=158.64 GlobalRate=190.31 Time=Mon Aug 24 08:31:21 2020
[xla:0](200) Loss=0.51948 Rate=178.82 GlobalRate=191.28 Time=Mon Aug 24 08:31:54 2020
[xla:0](300) Loss=0.50981 Rate=188.38 GlobalRate=192.42 Time=Mon Aug 24 08:32:27 2020
[xla:0](400) Loss=0.53478 Rate=190.21 GlobalRate=192.17 Time=Mon Aug 24 08:33:00 2020
[xla:0](500) Loss=0.42883 Rate=190.30 GlobalRate=191.81 Time=Mon Aug 24 08:33:34 2020
[xla:0](600) Loss=0.56005 Rate=192.33 GlobalRate=192.12 Time=Mon Aug 24 08:34:07 2020
[xla:0](700) Loss=0.46459 Rate=192.08 GlobalRate=192.09 Time=Mon Aug 24 08:34:40 2020
[xla:0](800) Loss=0.57102 Rate=192.75 GlobalRate=192.22 Time=Mon Aug 24 08:35:13 2020
[xla:0](900) Loss=0.52727 Rate=192.68 GlobalRate=192.27 Time=Mon Aug 24 08:35:47 2020
[xla:0](1000) Loss=0.49571 Rate=192.27 GlobalRate=192.24 Time=Mon Aug 24 08:36:20 2020
[xla:0](1100) Loss=0.53022 Rate=193.04 GlobalRate=192.36 Time=Mon Aug 24 08:36:53 2020
[xla:0](1200) Loss=0.52732 Rate=193.25 GlobalRate=192.45 Time=Mon Aug 24 08:37:26 2020
[xla:0](1300) Loss=0.53171 Rate=192.11 GlobalRate=192.36 Time=Mon Aug 24 08:37:59 2020
[xla:0](1400) Loss=0.55711 Rate=191.96 GlobalRate=192.33 Time=Mon Aug 24 08:38:33 2020
[xla:0](1500) Loss=0.47899 Rate=191.35 GlobalRate=192.23 Time=Mon Aug 24 08:39:06 2020
[xla:0](1600) Loss=0.49313 Rate=191.98 GlobalRate=192.24 Time=Mon Aug 24 08:39:40 2020
[xla:0](1700) Loss=0.49917 Rate=191.70 GlobalRate=192.20 Time=Mon Aug 24 08:40:13 2020
[xla:0](1800) Loss=0.45113 Rate=191.00 GlobalRate=192.11 Time=Mon Aug 24 08:40:47 2020
[xla:0](1900) Loss=0.55683 Rate=191.94 GlobalRate=192.13 Time=Mon Aug 24 08:41:20 2020
[xla:0](2000) Loss=0.50309 Rate=189.62 GlobalRate=191.92 Time=Mon Aug 24 08:41:54 2020
[xla:0](2100) Loss=0.43908 Rate=191.80 GlobalRate=191.99 Time=Mon Aug 24 08:42:27 2020
[xla:0](2200) Loss=0.55668 Rate=191.56 GlobalRate=191.96 Time=Mon Aug 24 08:43:00 2020
[xla:0](2300) Loss=0.57430 Rate=192.42 GlobalRate=192.00 Time=Mon Aug 24 08:43:34 2020
[xla:0](2400) Loss=0.51146 Rate=191.65 GlobalRate=191.97 Time=Mon Aug 24 08:44:07 2020
[xla:0](2500) Loss=0.53493 Rate=191.79 GlobalRate=191.96 Time=Mon Aug 24 08:44:40 2020
Finished training epoch 1
[xla:0](0) Acc=0.84375 Rate=0.00 GlobalRate=0.00 Time=Mon Aug 24 08:45:09 2020
[xla:0] Accuracy=88.10%
Finished test epoch 1, valid=88.10
[xla:0](0) Loss=0.51107 Rate=124.98 GlobalRate=124.97 Time=Mon Aug 24 08:45:14 2020
[xla:0](100) Loss=0.49826 Rate=164.86 GlobalRate=190.45 Time=Mon Aug 24 08:45:47 2020
[xla:0](200) Loss=0.51933 Rate=180.46 GlobalRate=190.65 Time=Mon Aug 24 08:46:21 2020
[xla:0](300) Loss=0.51143 Rate=187.83 GlobalRate=191.34 Time=Mon Aug 24 08:46:54 2020
[xla:0](400) Loss=0.53570 Rate=191.62 GlobalRate=192.04 Time=Mon Aug 24 08:47:27 2020
[xla:0](500) Loss=0.42714 Rate=191.76 GlobalRate=192.00 Time=Mon Aug 24 08:48:01 2020
[xla:0](600) Loss=0.53513 Rate=191.72 GlobalRate=191.95 Time=Mon Aug 24 08:48:34 2020
[xla:0](700) Loss=0.46284 Rate=192.41 GlobalRate=192.08 Time=Mon Aug 24 08:49:07 2020
[xla:0](800) Loss=0.52079 Rate=192.12 GlobalRate=192.06 Time=Mon Aug 24 08:49:40 2020
[xla:0](900) Loss=0.52980 Rate=192.37 GlobalRate=192.11 Time=Mon Aug 24 08:50:14 2020
[xla:0](1000) Loss=0.48813 Rate=192.72 GlobalRate=192.20 Time=Mon Aug 24 08:50:47 2020
[xla:0](1100) Loss=0.49315 Rate=191.80 GlobalRate=192.10 Time=Mon Aug 24 08:51:20 2020
[xla:0](1200) Loss=0.52646 Rate=192.34 GlobalRate=192.15 Time=Mon Aug 24 08:51:54 2020
[xla:0](1300) Loss=0.49502 Rate=191.69 GlobalRate=192.08 Time=Mon Aug 24 08:52:27 2020
[xla:0](1400) Loss=0.55274 Rate=190.96 GlobalRate=191.97 Time=Mon Aug 24 08:53:01 2020
[xla:0](1500) Loss=0.47684 Rate=191.06 GlobalRate=191.91 Time=Mon Aug 24 08:53:34 2020
[xla:0](1600) Loss=0.49258 Rate=190.94 GlobalRate=191.84 Time=Mon Aug 24 08:54:08 2020
[xla:0](1700) Loss=0.54605 Rate=190.86 GlobalRate=191.78 Time=Mon Aug 24 08:54:41 2020
[xla:0](1800) Loss=0.41191 Rate=191.76 GlobalRate=191.82 Time=Mon Aug 24 08:55:14 2020
[xla:0](1900) Loss=0.56704 Rate=190.91 GlobalRate=191.74 Time=Mon Aug 24 08:55:48 2020
[xla:0](2000) Loss=0.46631 Rate=191.89 GlobalRate=191.78 Time=Mon Aug 24 08:56:21 2020
[xla:0](2100) Loss=0.41694 Rate=191.58 GlobalRate=191.76 Time=Mon Aug 24 08:56:55 2020
[xla:0](2200) Loss=0.55897 Rate=192.14 GlobalRate=191.79 Time=Mon Aug 24 08:57:28 2020
[xla:0](2300) Loss=0.54048 Rate=191.94 GlobalRate=191.79 Time=Mon Aug 24 08:58:01 2020
[xla:0](2400) Loss=0.48976 Rate=193.10 GlobalRate=191.88 Time=Mon Aug 24 08:58:34 2020
[xla:0](2500) Loss=0.52112 Rate=192.27 GlobalRate=191.87 Time=Mon Aug 24 08:59:08 2020
Finished training epoch 2
[xla:0](0) Acc=0.92188 Rate=0.00 GlobalRate=0.00 Time=Mon Aug 24 08:59:37 2020
[xla:0] Accuracy=92.06%
Finished test epoch 2, valid=92.06
('DONE', 92.06349206349206)