#Introduction
从网上获取的合同模板，根据网上标注的分类，将合同分为24类。一部分数据没有分类，作为测试数据集。
使用清洗过的308.5MB，59266行标注数据作为训练集；将2400行数据作为验证集，验证集参与训练，24类每类100条。

训练过程如图1所示： 

![ALT](science/htfl-train.svg)


#BERT ACC:79.73%
#CNN  ACC:32.67%

#https://colab.research.google.com/drive/1SZ62rhHlP5x3w_XwXndbvicPiCxT44IM#scrollTo=HaQK5Xa2FhbK


2020-07-29 05:24:03.356237: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
2020-07-29 05:24:14.572131: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) bb2a8d5e78e0:47315
2020-07-29 05:24:14.623983: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) bb2a8d5e78e0:47315
2020-07-29 05:24:14.677123: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) bb2a8d5e78e0:47315
2020-07-29 05:24:14.731273: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) bb2a8d5e78e0:47315
2020-07-29 05:24:14.786327: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) bb2a8d5e78e0:47315
2020-07-29 05:24:14.838088: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) bb2a8d5e78e0:47315
2020-07-29 05:24:14.890492: I tensorflow/compiler/xla/xla_client/mesh_service.cc:234] Waiting to connect to client mesh master (300 seconds) bb2a8d5e78e0:47315
Loading train records for train...
2020-07-29 05:24:22.693020: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at bb2a8d5e78e0:47315
2020-07-29 05:24:23.730192: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at bb2a8d5e78e0:47315
2020-07-29 05:24:23.779363: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at bb2a8d5e78e0:47315
2020-07-29 05:24:23.850297: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at bb2a8d5e78e0:47315
2020-07-29 05:24:24.059430: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at bb2a8d5e78e0:47315
2020-07-29 05:24:24.290321: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at bb2a8d5e78e0:47315
2020-07-29 05:24:24.336066: I tensorflow/compiler/xla/xla_client/computation_client.cc:195] Fetching mesh configuration for worker tpu_worker:0 from mesh service at bb2a8d5e78e0:47315
10387it [00:05, 1786.91it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 2050.83it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1689.31it/s]
153 valid records loaded.
Loading train records for train...
[xla:0](0) Loss=2.83556 Rate=5.19 GlobalRate=5.19 Time=Wed Jul 29 05:27:42 2020
10387it [00:05, 1779.30it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 1999.45it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1669.96it/s]
153 valid records loaded.
Loading train records for train...
10387it [00:05, 1757.97it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 1982.40it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1626.62it/s]
153 valid records loaded.
Loading train records for train...
10387it [00:05, 1775.07it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 2046.40it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1656.55it/s]
153 valid records loaded.
Loading train records for train...
10387it [00:05, 1781.14it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 1980.23it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1640.68it/s]
153 valid records loaded.
Loading train records for train...
10387it [00:05, 1760.30it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 2008.49it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1637.44it/s]
153 valid records loaded.
Loading train records for train...
10387it [00:05, 1737.88it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 1984.08it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1594.13it/s]
153 valid records loaded.
Loading train records for train...
10387it [00:05, 1768.84it/s]
10387 training records loaded.
Loading train records for valid...
10387it [00:05, 1999.67it/s]
10387 train records loaded.
Loading valid records...
153it [00:00, 1617.48it/s]
153 valid records loaded.
[xla:0](100) Loss=2.77527 Rate=2.16 GlobalRate=0.14 Time=Wed Jul 29 05:51:14 2020
[xla:0](200) Loss=1.94137 Rate=4.01 GlobalRate=0.28 Time=Wed Jul 29 05:51:52 2020
[xla:0](300) Loss=2.42973 Rate=4.72 GlobalRate=0.40 Time=Wed Jul 29 05:52:31 2020
[xla:0](400) Loss=1.93031 Rate=4.88 GlobalRate=0.52 Time=Wed Jul 29 05:53:11 2020
[xla:0](500) Loss=2.41676 Rate=4.95 GlobalRate=0.64 Time=Wed Jul 29 05:53:51 2020
[xla:0](600) Loss=2.47447 Rate=4.96 GlobalRate=0.75 Time=Wed Jul 29 05:54:31 2020
Finished training epoch 0
[xla:0](0) Acc=0.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 05:54:53 2020
[xla:0] Accuracy=69.28%
Finished test epoch 0, valid=69.28
tcmalloc: large alloc 1182736384 bytes == 0x107f40000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
tcmalloc: large alloc 1786036224 bytes == 0x15f520000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
tcmalloc: large alloc 2690637824 bytes == 0x1f082a000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
saved model.
[xla:0](0) Loss=2.92952 Rate=7.91 GlobalRate=7.91 Time=Wed Jul 29 05:58:22 2020
[xla:0](100) Loss=1.93102 Rate=6.11 GlobalRate=4.93 Time=Wed Jul 29 05:59:02 2020
[xla:0](200) Loss=1.92953 Rate=5.48 GlobalRate=4.99 Time=Wed Jul 29 05:59:42 2020
[xla:0](300) Loss=2.42953 Rate=5.31 GlobalRate=5.06 Time=Wed Jul 29 06:00:20 2020
[xla:0](400) Loss=1.92953 Rate=5.22 GlobalRate=5.08 Time=Wed Jul 29 06:00:59 2020
[xla:0](500) Loss=2.41364 Rate=5.15 GlobalRate=5.09 Time=Wed Jul 29 06:01:38 2020
[xla:0](600) Loss=2.92854 Rate=5.19 GlobalRate=5.11 Time=Wed Jul 29 06:02:17 2020
Finished training epoch 1
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:02:36 2020
[xla:0] Accuracy=73.86%
Finished test epoch 1, valid=73.86
tcmalloc: large alloc 1786052608 bytes == 0x16f70c000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
tcmalloc: large alloc 2690629632 bytes == 0x1efc2a000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
saved model.
[xla:0](0) Loss=2.92953 Rate=8.22 GlobalRate=8.22 Time=Wed Jul 29 06:03:29 2020
[xla:0](100) Loss=2.40552 Rate=6.13 GlobalRate=4.76 Time=Wed Jul 29 06:04:12 2020
[xla:0](200) Loss=1.92953 Rate=5.45 GlobalRate=4.87 Time=Wed Jul 29 06:04:52 2020
[xla:0](300) Loss=2.42953 Rate=5.18 GlobalRate=4.92 Time=Wed Jul 29 06:05:32 2020
[xla:0](400) Loss=1.92953 Rate=5.07 GlobalRate=4.94 Time=Wed Jul 29 06:06:12 2020
[xla:0](500) Loss=1.94063 Rate=5.03 GlobalRate=4.95 Time=Wed Jul 29 06:06:52 2020
[xla:0](600) Loss=2.92336 Rate=5.00 GlobalRate=4.95 Time=Wed Jul 29 06:07:32 2020
Finished training epoch 2
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:07:52 2020
[xla:0] Accuracy=73.20%
Finished test epoch 2, valid=73.20
tcmalloc: large alloc 2690621440 bytes == 0x1ef42a000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
saved model.
[xla:0](0) Loss=2.45628 Rate=7.57 GlobalRate=7.57 Time=Wed Jul 29 06:09:37 2020
[xla:0](100) Loss=1.92953 Rate=6.06 GlobalRate=5.07 Time=Wed Jul 29 06:10:16 2020
[xla:0](200) Loss=1.92953 Rate=5.57 GlobalRate=5.15 Time=Wed Jul 29 06:10:54 2020
[xla:0](300) Loss=2.42953 Rate=5.36 GlobalRate=5.18 Time=Wed Jul 29 06:11:33 2020
[xla:0](400) Loss=1.92953 Rate=5.28 GlobalRate=5.19 Time=Wed Jul 29 06:12:11 2020
[xla:0](500) Loss=2.42952 Rate=5.25 GlobalRate=5.20 Time=Wed Jul 29 06:12:49 2020
[xla:0](600) Loss=2.45264 Rate=5.13 GlobalRate=5.17 Time=Wed Jul 29 06:13:29 2020
Finished training epoch 3
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:13:48 2020
[xla:0] Accuracy=75.82%
Finished test epoch 3, valid=75.82
tcmalloc: large alloc 2690646016 bytes == 0x1ef28a000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
saved model.
[xla:0](0) Loss=2.42953 Rate=7.63 GlobalRate=7.63 Time=Wed Jul 29 06:14:42 2020
[xla:0](100) Loss=1.92953 Rate=6.09 GlobalRate=5.07 Time=Wed Jul 29 06:15:22 2020
[xla:0](200) Loss=1.92953 Rate=5.43 GlobalRate=5.03 Time=Wed Jul 29 06:16:02 2020
[xla:0](300) Loss=2.42953 Rate=5.12 GlobalRate=4.99 Time=Wed Jul 29 06:16:43 2020
[xla:0](400) Loss=1.92953 Rate=5.03 GlobalRate=4.99 Time=Wed Jul 29 06:17:23 2020
[xla:0](500) Loss=2.42914 Rate=5.00 GlobalRate=4.98 Time=Wed Jul 29 06:18:03 2020
[xla:0](600) Loss=2.42953 Rate=4.99 GlobalRate=4.99 Time=Wed Jul 29 06:18:43 2020
Finished training epoch 4
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:19:03 2020
[xla:0] Accuracy=77.78%
Finished test epoch 4, valid=77.78
tcmalloc: large alloc 2690629632 bytes == 0x2519f4000 @  0x7f1164b732a4 0x591d67 0x4dd6a7 0x4dd77e 0x4e1d6d 0x4e1eab 0x4e0cf0 0x4e279b 0x4e210a 0x4e0d98 0x4e251b 0x4e2072 0x4e0cf0 0x4e251b 0x5eb622 0x4e0f43 0x4e251b 0x4e3386 0x5eb3d2 0x50a35c 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90 0x50a48d 0x50bfb4 0x507d64 0x509a90
saved model.
[xla:0](0) Loss=2.42964 Rate=5.77 GlobalRate=5.77 Time=Wed Jul 29 06:20:54 2020
[xla:0](100) Loss=1.92953 Rate=5.19 GlobalRate=4.81 Time=Wed Jul 29 06:21:36 2020
[xla:0](200) Loss=1.92953 Rate=5.21 GlobalRate=5.00 Time=Wed Jul 29 06:22:14 2020
[xla:0](300) Loss=2.42953 Rate=5.20 GlobalRate=5.07 Time=Wed Jul 29 06:22:53 2020
[xla:0](400) Loss=1.92953 Rate=5.21 GlobalRate=5.10 Time=Wed Jul 29 06:23:31 2020
[xla:0](500) Loss=1.93907 Rate=5.10 GlobalRate=5.09 Time=Wed Jul 29 06:24:11 2020
[xla:0](600) Loss=2.42977 Rate=5.02 GlobalRate=5.07 Time=Wed Jul 29 06:24:51 2020
Finished training epoch 5
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:25:11 2020
[xla:0] Accuracy=76.47%
Finished test epoch 5, valid=76.47
saved model.
[xla:0](0) Loss=2.43029 Rate=5.88 GlobalRate=5.88 Time=Wed Jul 29 06:28:09 2020
[xla:0](100) Loss=1.92953 Rate=5.25 GlobalRate=4.83 Time=Wed Jul 29 06:28:50 2020
[xla:0](200) Loss=1.92953 Rate=5.20 GlobalRate=5.00 Time=Wed Jul 29 06:29:29 2020
[xla:0](300) Loss=2.42953 Rate=5.19 GlobalRate=5.06 Time=Wed Jul 29 06:30:08 2020
[xla:0](400) Loss=1.92953 Rate=5.20 GlobalRate=5.10 Time=Wed Jul 29 06:30:46 2020
[xla:0](500) Loss=1.92954 Rate=5.20 GlobalRate=5.11 Time=Wed Jul 29 06:31:24 2020
[xla:0](600) Loss=2.42953 Rate=5.20 GlobalRate=5.13 Time=Wed Jul 29 06:32:03 2020
Finished training epoch 6
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:32:22 2020
[xla:0] Accuracy=73.86%
Finished test epoch 6, valid=73.86
saved model.
[xla:0](0) Loss=2.42953 Rate=7.07 GlobalRate=7.07 Time=Wed Jul 29 06:35:29 2020
[xla:0](100) Loss=1.92953 Rate=5.71 GlobalRate=4.81 Time=Wed Jul 29 06:36:11 2020
[xla:0](200) Loss=1.92953 Rate=5.28 GlobalRate=4.90 Time=Wed Jul 29 06:36:51 2020
[xla:0](300) Loss=2.43158 Rate=5.11 GlobalRate=4.93 Time=Wed Jul 29 06:37:31 2020
[xla:0](400) Loss=1.92953 Rate=5.08 GlobalRate=4.96 Time=Wed Jul 29 06:38:11 2020
[xla:0](500) Loss=1.92953 Rate=5.14 GlobalRate=5.01 Time=Wed Jul 29 06:38:49 2020
[xla:0](600) Loss=2.42953 Rate=5.17 GlobalRate=5.04 Time=Wed Jul 29 06:39:28 2020
Finished training epoch 7
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:39:47 2020
[xla:0] Accuracy=74.51%
Finished test epoch 7, valid=74.51
saved model.
[xla:0](0) Loss=2.42914 Rate=4.59 GlobalRate=4.59 Time=Wed Jul 29 06:40:42 2020
[xla:0](100) Loss=1.92953 Rate=4.87 GlobalRate=5.05 Time=Wed Jul 29 06:41:21 2020
[xla:0](200) Loss=1.92953 Rate=5.07 GlobalRate=5.13 Time=Wed Jul 29 06:42:00 2020
[xla:0](300) Loss=2.42953 Rate=5.13 GlobalRate=5.14 Time=Wed Jul 29 06:42:38 2020
[xla:0](400) Loss=1.92953 Rate=5.03 GlobalRate=5.09 Time=Wed Jul 29 06:43:19 2020
[xla:0](500) Loss=1.92953 Rate=4.97 GlobalRate=5.06 Time=Wed Jul 29 06:43:59 2020
[xla:0](600) Loss=2.51508 Rate=4.99 GlobalRate=5.05 Time=Wed Jul 29 06:44:39 2020
Finished training epoch 8
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:44:59 2020
[xla:0] Accuracy=73.20%
Finished test epoch 8, valid=73.20
saved model.
[xla:0](0) Loss=1.95194 Rate=8.11 GlobalRate=8.11 Time=Wed Jul 29 06:47:48 2020
[xla:0](100) Loss=1.92953 Rate=6.27 GlobalRate=5.06 Time=Wed Jul 29 06:48:27 2020
[xla:0](200) Loss=1.92953 Rate=5.65 GlobalRate=5.15 Time=Wed Jul 29 06:49:06 2020
[xla:0](300) Loss=2.42953 Rate=5.39 GlobalRate=5.17 Time=Wed Jul 29 06:49:44 2020
[xla:0](400) Loss=1.92953 Rate=5.29 GlobalRate=5.19 Time=Wed Jul 29 06:50:22 2020
[xla:0](500) Loss=1.92953 Rate=5.25 GlobalRate=5.19 Time=Wed Jul 29 06:51:00 2020
[xla:0](600) Loss=2.42953 Rate=5.24 GlobalRate=5.20 Time=Wed Jul 29 06:51:39 2020
Finished training epoch 9
[xla:0](0) Acc=1.00000 Rate=0.00 GlobalRate=0.00 Time=Wed Jul 29 06:51:58 2020
[xla:0] Accuracy=75.16%
Finished test epoch 9, valid=75.16
saved model.
('DONE', 0, 75.16339869281046)
saved model.

#55.48%