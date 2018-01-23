# 概要

圧縮されたJpeg画像の復元

# 動作環境

- Ubuntu 16.04.3 LTS ($ cat /etc/issue)
- Python 3.5.2 ($ python3 -V)
- chainer 3.2 ($ pip3 show chainer | grep Ver)
- numpy 1.13.3 ($ pip3 show numpy | grep Ver)
- cupy 2.2
- opencv-python 3.4.0.12

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 2 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
```

## ファイル

- ./Lib/myfunc.py
  - 便利機能
- ./Lib/network.py
  - jpegcompのネットワーク部分
- ./Tool/dot2png.py
  - dot言語で記述されたファイルをPNG形式に変換する
- ./Tool/npz2jpg.py
  - 作成したデータセット（.npz）の中身を画像として出力する
- ./Tool/plot_diff.py
  - logファイルの複数比較
- ./create_dataset.py
  - 画像を読み込んでデータセットを作成する
- ./predict.py
  - モデルとモデルパラメータを利用して推論実行する
- ./train.py
  - '学習メイン部'
- ./FontData
  - 学習とテストに使用する画像データなど（サイズが大きいので注意）

# チュートリアル

## データセットを作成する

### 実行

```console
$ ./create_dataset.py ./FontData/font_0[0-1]*
```

### 端末の確認
以下のとおりであれば正常に実行できている

```console
not import cupy
------------------------------
channel:	1
img_size:	32
jpeg[2]:
	./FontData/font_00.bmp
	./FontData/font_01.bmp
out_path:	./result/
quality:	5
round:	1000
train_per_all:	0.85
------------------------------
train comp: (6800, 32, 32)
      raw:  (6800, 32, 32)
test comp:  (1200, 32, 32)
     raw:   (1200, 32, 32)
```
`not import cupy`はcupyをインストールしていない場合に表示される

### 生成物の確認

resultフォルダが作成され、その中にtest.npzとtrain.npzが生成されていればOK

## 学習する

### 実行

```console
$ ./train.py
```

### 端末の確認

```console
not import cupy
------------------------------
actfun_1:	relu
actfun_2:	sigmoid
batchsize:	200
epoch:	10
frequency:	-1
gpu_id:	-1
in_path:	./result/
layer_num:	3
lossfun:	mse
only_check:	False
optimizer:	adam
out_path:	./result/
plot:	True
resume:
unit:	4
------------------------------
Activation func: relu
Activation func: sigmoid
[Network info]
  Unit:	4
  Out:	1
  Layer:	3
  Act Func:	relu, sigmoid
Loss func: mean_squared_error
Optimizer: Adam optimizer
test  (comp/raw): (1200, 32, 32)/(1200, 32, 32)
train (comp/raw): (6800, 32, 32)/(6800, 32, 32)
epoch       main/loss   validation/main/loss  elapsed_time
1           0.208491    0.119774              32.4628
2           0.174672    0.166305              65.61
3           0.158801    0.153584              97.5322
4           0.146055    0.148234              129.836
5           0.13466     0.138804              162.696
6           0.123279    0.123111              195.181
7           0.113303    0.113019              227.963
8           0.10562     0.101799              259.506
9           0.0994183   0.0968083             291.633
10          0.0940017   0.0894627             323.381
```

### 生成物の確認

resultフォルダ中に*.log、*_graph.dot、*_plot.png、*.snapshot、*.modelが生成されていればOK

## 学習で作成されたモデルを使用する

### 実行

```console
$ ./predict.py ./result/*.model ./result/*.json ./FontData/The_Night*
```

### 端末の確認

```console
not import cupy
------------------------------
batch:	100
gpu:	-1
img_size:	32
jpeg[2]:
	./FontData/The_Night_of_the_Milky_Way_Train_ch2.PNG
	./FontData/The_Nighthawk_Star_op.PNG
model:	./result/180122-153747.model
out_path:	./result/
param:	./result/180122-153747.json
quality:	5
------------------------------
Activation func: relu
Activation func: sigmoid
[Network info]
  Unit:	4
  Out:	1
  Layer:	3
  Act Func:	relu, sigmoid
model read
```

### 生成物の確認

resultフォルダ中にcomp-*.jpgファイルが生成されていればOK

# その他の機能

## 生成されたデータの削除

```console
$ ./clean_all.sh
```

## ハイパーパラメータを変更して検証

```console
$ ./auto_train.sh
```

## Dotファイルの画像化

```console
$ ./dot2png.py ./result/*.dot
```
