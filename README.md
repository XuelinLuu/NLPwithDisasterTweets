这是kaggle比赛的尝试，是一个nlp相关的入门级项目，对数据进行处理，然后使用bert模型进行训练，在测试集上得到最终结果，保存在result.csv文件中
- kaggle地址：https://www.kaggle.com/c/nlp-getting-started
- datasets
  - 保存项目所需数据集
- models
  - 保存训练所得模型
- results
  - 保存模型在测试集上的结果
- src
  - config
    - 保存项目相关配置文件
  - dataset
    - 对数据进行处理，清洗数据，并将数据转换为模型需要的数据
  - engine
    - 损失函数，训练函数和测试函数
  - model
    - 项目模型
  - run_train
    - 进行模型训练
  - run_test
    - 进行模型测试