# SMP-CAIL2020：论辩挖掘

本项目为 **中国法研杯司法人工智能挑战赛（CAIL2020）** 论辩挖掘赛道参考代码和模型。

主要包含两个基线模型：BERT和RNN。

### 0. 预处理

#### 0.0 下载本项目

```
git clone https://github.com/gaoyixu/CAIL2020-Argument-Mining.git
```

#### 0.1 数据集

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。

本项目中只使用了

`SMP-CAIL2020-train.csv`： 包含了2449对裁判文书中的互动论点对。分别包含以下维度：

  - `id`： 论点对id
  - `text_id`： 裁判文书id
  - `sc`： 论点对中诉方论点
  - `A/B/C/D/E`： 给出的五句候选辩方论点
  - `answer`： 辩方正确论点

划分训练集、验证集：

```
python prepare.py
```

#### 0.2 下载BERT模型（pytorch版本）

下载中文预训练BERT模型存放于`model/bert`和`model/bert/bert-base-chinese`目录

中文预训练BERT模型包含三个文件：

1. [`config.json`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json) 
2. [`pytorch_model.bin`](https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin)
3. [`vocab.txt`](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt) 

初始文件目录：

```
├── config
│   ├── bert_config.json
│   └── rnn_config.json
├── data
│   ├── SMP-CAIL2020-train.csv
│   ├── train.csv
│   └── valid.csv
├── model
│   ├── bert
│   │   ├── bert-base-chinese
│   │   │   ├── config.json
│   │   │   └── pytorch_model.bin
│   │   └── vocab.txt
│   └── rnn
├── __init__.py
├── data.py
├── evaluate.py
├── main.py
├── model.py
├── prepare.py
├── result.py
├── test.py
├── train.py
├── utils.py
└── vocab.py
```

### 1. 训练

#### 1.1 BERT训练

采用4张1080Ti训练，训练参数可在`config/bert_config.json`中调整。

```
python -m torch.distributed.launch train.py --config_file 'config/bert_config.json'
```

<div align = "center">
  <img src="images/bert_train.png" width = "50%"/>
</div>

#### 1.2 RNN训练

采用1张1080Ti训练，训练参数可在`config/rnn_config.json`中调整。

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch train.py --config_file 'config/rnn_config.json'
```

<div align = center>
  <img src="images/rnn_train.png" width = "50%"/>
</div>

#### 1.3 训练成果

训练完成后文件目录：

`config`中包含模型训练参数。

`log`中包含模型每个epoch的Accuracy，F1 Score和每步loss的记录数据。

`model`中包含每个epoch训练后的模型和验证集上F1 Score最高的模型`model.bin`。

```
├── config
│   ├── bert_config.json
│   └── rnn_config.json
├── data
│   ├── SMP-CAIL2020-train.csv
│   ├── train.csv
│   └── valid.csv
├── log
│   ├── BERT-epoch.csv
│   ├── BERT-step.csv
│   ├── RNN-epoch.csv
│   └── RNN-step.csv
├── model
│   ├── bert
│   │   ├── BERT
│   │   │   ├── bert-1.bin
│   │   │   ├── bert-2.bin
│   │   │   ├── bert-3.bin
│   │   │   ├── bert-4.bin
│   │   │   ├── bert-5.bin
│   │   │   ├── bert-6.bin
│   │   │   ├── bert-7.bin
│   │   │   ├── bert-8.bin
│   │   │   ├── bert-9.bin
│   │   │   └── bert-10.bin
│   │   ├── bert-base-chinese
│   │   │   ├── config.json
│   │   │   └── pytorch_model.bin
│   │   └── vocab.txt
│   └── rnn
│       ├── model.bin
│       ├── RNN
│       │   ├── rnn-1.bin
│       │   ├── rnn-2.bin
│       │   ├── rnn-3.bin
│       │   ├── rnn-4.bin
│       │   ├── rnn-5.bin
│       │   ├── rnn-6.bin
│       │   ├── rnn-7.bin
│       │   ├── rnn-8.bin
│       │   ├── rnn-9.bin
│       │   └── rnn-10.bin
│       └── vocab.txt
├── __init__.py
├── data.py
├── evaluate.py
├── main.py
├── model.py
├── prepare.py
├── result.py
├── test.py
├── train.py
├── utils.py
└── vocab.py
```

### 2. 测试

`in_file`为待测试文件，`out_file`为输出文件。

#### 2.1 BERT测试

```
python main.py --model_config 'config/bert_config.json' \
               --in_file 'SMP-CAIL2020-test1.csv' \
               --out_file 'bert-submission-test-1.csv'
```

#### 2.2 RNN测试

```
python main.py --model_config 'config/rnn_config.json' \
               --in_file 'data/SMP-CAIL2020-test1.csv' \
               --out_file 'rnn-submission-test-1.csv'
```

### RESULT
202000619,   BERT model,    TEST result: 0.67-0.682
Epoch: 1, train_acc: 0.726073, train_f1: 0.725676, valid_acc: 0.760000, valid_f1: 0.768889,

202000622,   BERT+1-7CNN,   TEST result: 0.696
1,0.7909350755410371,0.790836697236448,0.838,0.8352161478089404
2,0.7913434054716211,0.7912636458834423,0.838,0.8352161478089404

202000622,   BERT+1-14CNN,   TEST result: 0.669 (1 epoch)
train_acc: 0.770110, train_f1: 0.769986, valid_acc: 0.788000, valid_f1: 0.787216,

202000622,   BERT+1-14CNN,   TEST result: 0.669  -  0.75 (3 epoch)
1,0.8836259697835851,0.883557016059752,0.918,0.9195225562637497
2,0.8962841976316864,0.8963458000980952,0.946,0.9452992509238977
3,0.9309922417313189,0.9310764659907621,0.972,0.971331180849851   TEST:0.75
4,0.9403838301347489,0.940379133280375,0.978,0.9778776608845254
