## 项目介绍

本项目使用TensorFlow搭建了语音识别模型。

依赖环境：

* python 3.5
* tensorflow 1.4
* python_speech_features 用于处理语音信号得到 mfcc
* scipy 用于读取wav文件
* sphfile 用于将TIMIT库中的 Nist Sphere 格式的文件转换为wav方便处理




## 文件介绍

1. `convnet.py`

   定义了一个CNN+DNN网络，输入数据是不同人读数字的音频，经过处理变成MFCC后看作一个单通道图片，使用类似图像识别的方法，对输出的数字进行预测。

2. `params.py`

   convnet 中用到的激活函数。

3. `timit_ctc.py`

   文件中定义了bidirectional LSTM + CTC，训练数据为TIMIT语音库，将输入的完整音频处理为MFCC序列后，作为bidirectional LSTM网络的输入，得到的网络输出结果使用CTC解码，得到输出的字母序列。CTC的具体流程和原理请参考[Sequence ModelingWith CTC](https://distill.pub/2017/ctc/).

4. `preprocess.py`

   用于timit数据集的预处理。

5. `data_utils.py` 

   该文件中主要定义了一些数据处理相关的函数，比如wav文件读取，转换成mfcc，batch的生成等等。

6. `data` 文件夹

   内部存放相关训练数据，`spoken_numbers_pcm` 中存放的为数字发音的音频，`TIMIT` 文件夹中为TIMIT数据集。



## TIMIT数据集简要介绍

> TIMIT全称The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus, 是由德州仪器(TI)、麻省理工学院(MIT)和坦福研究院(SRI)合作构建的声学-音素连续语音语料库。TIMIT数据集的语音采样频率为16kHz，一共包含6300个句子，由来自美国八个主要方言地区的630个人每人说出给定的10个句子，所有的句子都在音素级别(phone level)上进行了手动分割，标记。70%的说话人是男性；大多数说话者是成年白人。

更加具体的介绍请参考TIMIT数据集中的README文件。

## 使用方法

### convnet

与模型相关的参数位于`convnet.py`代码前部与`params.py`中，直接运行即可。

```shell
python convnet.py
```

程序运行过程中将会将统计信息写入到当前目录的`tmp/train`和 `tmp/vali` 中，使用tensorboard可以进行查看。

```shell
tensorboard --logdir ./tmp 
```



### timit_ctc

1. 首先将TIMIT数据集解压缩到`data`文件夹中，命名为TIMIT。

2. 运行`preprocess.py`（只有第一次运行程序的时候需要运行）

   ```shell
   python preprocess.py
   ```

3. 与模型相关的参数位于`timit_ctc`代码前部，调整完成后运行即可

   ```shell
   python timit_ctc.py
   ```

   运行过程中默认输出模型的统计信息，每10个step输出一次预测结果，每5个epoch储存一次模型信息，存在`./tmp/checkpoint`文件夹中，模型训练过程中将会首先检查有没有checkpoint文件，有的话将会继续训练，如果想要从头开始请删除对应文件夹。训练过程的统计信息也保存在了`./tmp/summary`文件夹中，可以使用tensorboard查看。

   ```shell
   tensorboard --logdir ./tmp/summary
   ```

   ​