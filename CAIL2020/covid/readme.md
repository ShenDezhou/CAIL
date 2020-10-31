# Introduction

compare between BERT-LARGE with RoBERTa-3-LARGE-ext model.

## 1.BERT-LARGE model
```
[xla:0](100) Loss=1.16723 Rate=0.07 GlobalRate=0.05 Time=Thu Sep 24 07:41:56 2020
[xla:0](200) Loss=1.43893 Rate=4.44 GlobalRate=0.10 Time=Thu Sep 24 07:42:23 2020
[xla:0](300) Loss=0.99112 Rate=6.50 GlobalRate=0.14 Time=Thu Sep 24 07:42:49 2020
[xla:0](400) Loss=0.90796 Rate=6.86 GlobalRate=0.19 Time=Thu Sep 24 07:43:17 2020
[xla:0](500) Loss=0.97639 Rate=7.10 GlobalRate=0.23 Time=Thu Sep 24 07:43:44 2020
[xla:0](600) Loss=1.26334 Rate=7.27 GlobalRate=0.28 Time=Thu Sep 24 07:44:12 2020
[xla:0](700) Loss=0.87083 Rate=7.31 GlobalRate=0.32 Time=Thu Sep 24 07:44:39 2020
[xla:0](800) Loss=0.75612 Rate=7.25 GlobalRate=0.37 Time=Thu Sep 24 07:45:07 2020
[xla:0](900) Loss=0.76049 Rate=7.42 GlobalRate=0.41 Time=Thu Sep 24 07:45:33 2020
[xla:0](1000) Loss=0.74393 Rate=7.29 GlobalRate=0.45 Time=Thu Sep 24 07:46:01 2020
[xla:0](1100) Loss=0.74535 Rate=7.22 GlobalRate=0.49 Time=Thu Sep 24 07:46:29 2020
[xla:0](1200) Loss=0.74383 Rate=7.15 GlobalRate=0.53 Time=Thu Sep 24 07:46:57 2020
[xla:0](1300) Loss=0.74376 Rate=7.31 GlobalRate=0.58 Time=Thu Sep 24 07:47:24 2020
[xla:0](1400) Loss=0.74370 Rate=7.48 GlobalRate=0.62 Time=Thu Sep 24 07:47:50 2020
[xla:0](1500) Loss=0.74370 Rate=7.47 GlobalRate=0.66 Time=Thu Sep 24 07:48:17 2020
[xla:0](1600) Loss=0.74368 Rate=7.39 GlobalRate=0.70 Time=Thu Sep 24 07:48:44 2020
[xla:0](1700) Loss=0.74367 Rate=7.44 GlobalRate=0.74 Time=Thu Sep 24 07:49:11 2020
[xla:0](1800) Loss=0.74367 Rate=7.36 GlobalRate=0.77 Time=Thu Sep 24 07:49:38 2020
[xla:0](1900) Loss=0.74367 Rate=7.45 GlobalRate=0.81 Time=Thu Sep 24 07:50:05 2020
[xla:0](2000) Loss=0.74367 Rate=7.58 GlobalRate=0.85 Time=Thu Sep 24 07:50:31 2020
[xla:0](2100) Loss=0.74367 Rate=7.58 GlobalRate=0.89 Time=Thu Sep 24 07:50:57 2020
Finished training epoch 0
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:51:29 2020
[xla:0](100) Acc=0.98515 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:52:04 2020
[xla:0](200) Acc=0.98756 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:52:28 2020
[xla:0](300) Acc=0.98837 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:52:51 2020
[xla:0](400) Acc=0.99002 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:53:15 2020
[xla:0](500) Acc=0.98503 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:53:38 2020
[xla:0](600) Acc=0.98752 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:54:02 2020
[xla:0](700) Acc=0.98859 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:54:26 2020
[xla:0](800) Acc=0.98814 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:54:49 2020
[xla:0](900) Acc=0.98890 Rate=0.00 GlobalRate=0.00 Time=Thu Sep 24 07:55:13 2020
[xla:0] Accuracy=98.95%
Finished test epoch 0, valid=98.95
('DONE', 98.95)
saved model.
```


# 2.RoBERTa-3-Large-Ext model
```
[xla:0](100) Loss=1.36708 Rate=0.29 GlobalRate=0.05 Time=Fri Sep 25 07:46:23 2020
[xla:0](200) Loss=1.38933 Rate=29.32 GlobalRate=0.10 Time=Fri Sep 25 07:46:27 2020
[xla:0](300) Loss=0.90580 Rate=41.29 GlobalRate=0.15 Time=Fri Sep 25 07:46:31 2020
[xla:0](400) Loss=0.90670 Rate=46.25 GlobalRate=0.20 Time=Fri Sep 25 07:46:36 2020
[xla:0](500) Loss=1.20364 Rate=47.92 GlobalRate=0.24 Time=Fri Sep 25 07:46:40 2020
[xla:0](600) Loss=1.11587 Rate=48.59 GlobalRate=0.29 Time=Fri Sep 25 07:46:44 2020
[xla:0](700) Loss=1.03590 Rate=48.93 GlobalRate=0.34 Time=Fri Sep 25 07:46:48 2020
[xla:0](800) Loss=0.77885 Rate=48.77 GlobalRate=0.39 Time=Fri Sep 25 07:46:52 2020
[xla:0](900) Loss=0.82693 Rate=49.07 GlobalRate=0.44 Time=Fri Sep 25 07:46:56 2020
[xla:0](1000) Loss=0.74579 Rate=49.28 GlobalRate=0.49 Time=Fri Sep 25 07:47:00 2020
[xla:0](1100) Loss=0.75016 Rate=49.32 GlobalRate=0.53 Time=Fri Sep 25 07:47:04 2020
[xla:0](1200) Loss=1.02060 Rate=49.23 GlobalRate=0.58 Time=Fri Sep 25 07:47:08 2020
[xla:0](1300) Loss=0.74703 Rate=49.31 GlobalRate=0.63 Time=Fri Sep 25 07:47:12 2020
[xla:0](1400) Loss=0.74613 Rate=49.35 GlobalRate=0.68 Time=Fri Sep 25 07:47:16 2020
[xla:0](1500) Loss=0.76008 Rate=49.55 GlobalRate=0.73 Time=Fri Sep 25 07:47:20 2020
[xla:0](1600) Loss=0.74436 Rate=48.80 GlobalRate=0.77 Time=Fri Sep 25 07:47:24 2020
[xla:0](1700) Loss=0.74433 Rate=49.21 GlobalRate=0.82 Time=Fri Sep 25 07:47:28 2020
[xla:0](1800) Loss=0.74404 Rate=49.29 GlobalRate=0.87 Time=Fri Sep 25 07:47:32 2020
[xla:0](1900) Loss=0.74386 Rate=49.39 GlobalRate=0.92 Time=Fri Sep 25 07:47:37 2020
[xla:0](2000) Loss=0.74873 Rate=49.71 GlobalRate=0.96 Time=Fri Sep 25 07:47:41 2020
[xla:0](2100) Loss=0.74387 Rate=49.62 GlobalRate=1.01 Time=Fri Sep 25 07:47:45 2020
Finished training epoch 0
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:47:52 2020
[xla:0](100) Acc=0.97030 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:47:58 2020
[xla:0](200) Acc=0.96766 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:03 2020
[xla:0](300) Acc=0.96678 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:07 2020
[xla:0](400) Acc=0.96758 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:11 2020
[xla:0](500) Acc=0.96108 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:16 2020
[xla:0](600) Acc=0.96173 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:20 2020
[xla:0](700) Acc=0.96362 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:25 2020
[xla:0](800) Acc=0.96317 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:29 2020
[xla:0](900) Acc=0.96337 Rate=0.00 GlobalRate=0.00 Time=Fri Sep 25 07:48:33 2020
[xla:0] Accuracy=96.35%
Finished test epoch 0, valid=96.35
('DONE', 96.35)
```
Acc drops on 2.6%

# COST
1. BERT-large
15411713558ns
13870042109ns
14300081014ns
12136016172ns
12392196777ns
12904527936ns

2. RoBERTa-3-Large-ext
2161661649ns
1966810849ns
2085638407ns
2046428213ns
2026669033ns
2279752473ns

同比下降-84.488%，209ms（原1350ms）

# CIFAR-10数据 data和filenmaes 示例:
Pandas(labels=9, data=array([142, 143, 144, ...,  52,  44,  38], dtype=uint8), filenames=b'moving_van_s_000051.png')
Pandas(labels=5, data=array([133, 162, 168, ...,  51,  50,  60], dtype=uint8), filenames=b'pekinese_s_000458.png')
Pandas(labels=1, data=array([240, 233, 238, ..., 132, 132, 129], dtype=uint8), filenames=b'wagon_s_001343.png')
Pandas(labels=1, data=array([255, 252, 253, ..., 234, 232, 232], dtype=uint8), filenames=b'automobile_s_002395.png')
Pandas(labels=9, data=array([  1,   6,  23, ..., 151, 153, 155], dtype=uint8), filenames=b'aerial_ladder_truck_s_001180.png')
Pandas(labels=1, data=array([129, 128, 130, ..., 185, 184, 187], dtype=uint8), filenames=b'police_cruiser_s_001389.png')
Pandas(labels=9, data=array([154, 154, 184, ...,  52,  30,  32], dtype=uint8), filenames=b'lorry_s_001596.png')
Pandas(labels=2, data=array([107,  88,  80, ..., 139,  62, 106], dtype=uint8), filenames=b'sparrow_s_001912.png')
Pandas(labels=4, data=array([197, 203, 197, ...,  97,  98,  87], dtype=uint8), filenames=b'mule_deer_s_001733.png')
Pandas(labels=9, data=array([255, 255, 255, ..., 255, 255, 255], dtype=uint8), filenames=b'delivery_truck_s_001529.png')
Pandas(labels=1, data=array([ 34,  41,  49, ..., 175, 174, 174], dtype=uint8), filenames=b'coupe_s_001573.png')
Pandas(labels=9, data=array([64, 59, 68, ..., 89, 91, 85], dtype=uint8), filenames=b'lorry_s_000018.png')
Pandas(labels=8, data=array([224, 224, 224, ..., 213, 212, 195], dtype=uint8), filenames=b'banana_boat_s_001615.png')