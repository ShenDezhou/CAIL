#Introduction
从网上获取的合同模板，根据网上标注的分类，将合同分为24类。一部分数据没有分类，作为测试数据集。
使用清洗过的308.5MB，59266行标注数据作为训练集；将2400行数据作为验证集，验证集参与训练，24类每类100条。

训练过程如图1所示： 

![ALT](science/htfl-train.svg)


#BERT ACC:79.73%
#CNN  ACC:32.67%

#https://colab.research.google.com/drive/1SZ62rhHlP5x3w_XwXndbvicPiCxT44IM#scrollTo=HaQK5Xa2FhbK


#BERT ACC:71.89%
#BERT LARGE:75.16%

# Docker large:
`docker run -d -it -v/mnt/data/htfldocker/data:/workspace/data -v /mnt/data/wwm_large_ext:/workspace/wwm_large_ext -v /mnt/data/htfl/model/bertl:/workspace/model/bert/  --gpus all --network host contractclassification:1.0 python3 torch_server.py -p 58081`

# 默认加载大模型