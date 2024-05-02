# 用于Fashion-MNIST数据分类任务的三层感知机
### 1.  代码模块介绍

本实验不依赖现有的深度学习框架如PyTorch或TensorFlow等，仅使用NumPy库，自主构建了一个三层神经网络分类器。实验使用Fashion-MNIST数据集，最终提交的代码中包含了**模型**、**训练**、**测试**和**参数查找**四个部分，进行了模块化设计。 其中，主要代码模块为：

**processData.py**：数据预处理和数据导入模块，若数据不存在（文件夹"./data/"不存在），则会自动进行数据下载，创建目录"./data/FashionMNIST/raw/"，并将数据保存到这里。

**MyNN.py**：允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度，实现了SGD优化器、学习率权重衰减、交叉熵损失和L2正则化，并能根据验证集指标自动保存最优的模型权重。**paramSearch.py**：使用网格搜索法调节学习率、学习率权重衰减强度、各隐藏层大小、L2正则化强度（weight_dacay）等超参数

**train.py**：用最优超参数训练模型，并保存到*model.pkl*中。

**test.py**：导入训练好的模型，输出在测试集上的分类准确率等指标。

**paramVisualize.ipynb**：导入训练好的模型，进行网格参数热力图可视化及各网络层输出PCA降维后可视化。

### 2.  代码调用流程

#### 2.1 超参数调优

```
python3 paramSearch.py
```
本步骤过于耗时，40余种超参数组合完整搜索，在单进程下耗时7h，**建议仅任用一种超参数组合进行测试**

具体地，在实验中对如下超参数组合中，进行了网格搜索：

> lr：[1e-1, 1e-2]
> hidden1_dim：[64, 128, 256] 
> hidden2_dim：[64, 128, 256] 
> weight_decay：[1, 1e-1, 1e-3]  

根据在交叉验证表现，得到的**best parameters**为：

> lr=0.1
>
> hidden1_dim=256
>
> hidden2_dim=256
>
> weight_decay=0.001  


各模型在训练集和测试集上的具体表现保存在了 *search_parameters.csv*.

#### 2. 2 用最优超参数的模型进行Train

```
python3 train.py
```
完成训练后，模型被保存在 *model.pkl*.

#### 2.3 读取保存好的模型进行Test

```
python3 test.py
```
读取*model.pkl*，进行训练；程序会输出对各类别的预测精度 

### 3.  训练好的模型权重

- 可直接读取*model.pkl文件*；或访问下面的百度网盘链接下载（提取码6026）
- [模型参数-百度网盘链接](https://pan.baidu.com/s/1rx2ALD49NiAm9BLkOOaCQQ?pwd=6026)
