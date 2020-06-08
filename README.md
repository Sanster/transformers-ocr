参考链接：
- https://github.com/macanv/BERT-BiLSTM-CRF-NER
- https://github.com/huggingface/transformers/tree/master/examples/token-classification

## 启动训练
```bash
python -X utf8 run_ner.py config.json
```

## 测试
```bash
python predict.py

Input: 在福特的帮助下，阿瑟·登特在地球被毁灭前的最后一刻搭上了一艘路过地球的外星人的太空船，远离这个即将毁灭的伤心地，开始了一段充满惊奇的星河探险
('PER', 2, 3): 福特
('PER', 9, 13): 阿瑟·登特
('LOC', 15, 16): 地球
('LOC', 33, 34): 地球
```

## 结果
|结构|dev recall|dev precision|dev f1|test recall|test precision|test f1|
|----|----------|-------------|------|-----------|-------------|-------|
|[BertForTokenClassification](https://github.com/huggingface/transformers/blob/1b5820a56540a2096daeb43a0cd8247c8c94a719/src/transformers/modeling_bert.py#L1296)(a linear layer on top of the Bert hidden-states)|0.959|0.951|0.955|0.956|0.942|0.949|
|Bert + BiLSTM|
|Bert + BiLSTM + CRF|

## 训练数据
训练数据来自：https://github.com/zjy-ucas/ChineseNER

把训练集(train.txt)，验证集(dev.txt)和测试集合(test.txt)放在同一个文件夹，文件内容格式如下，多个样本之间以空白行分隔：
```
海 O
钓 O
比 O
赛 O
地 O
点 O
在 O
厦 B-LOC
门 I-LOC
与 O
金 B-LOC
门 I-LOC
之 O
间 O
的 O
海 O
域 O
。 O

相 O
比 O
之 O
下 O
， O
青 B-ORG
岛 I-ORG
海 I-ORG
牛 I-ORG
队 I-ORG
和 O
广 B-ORG
州 I-ORG
松 I-ORG
日 I-ORG
队 I-ORG
```

如何从训练集生成 `labels.txt`:
```bash
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
```

