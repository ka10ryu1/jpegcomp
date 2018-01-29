# 概要

圧縮されたJpeg画像の復元

## 学習結果（5%に圧縮した画像を復元）

圧縮画像

復元画像

# 動作環境

- Ubuntu 16.04.3 LTS ($ cat /etc/issue)
- Python 3.5.2 ($ python3 -V)
- chainer 3.2 ($ pip3 show chainer | grep Ver)
- numpy 1.13.3 ($ pip3 show numpy | grep Ver)
- cupy 2.2 ($ pip3 show cupy | grep Ver)
- opencv-python 3.4.0.12 ($ pip3 show opencv-python | grep Ver)

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 2 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
```

## ファイル

```console
├── FontData
│   ├── The_Night_of_the_Milky_Way_Train_ch2.PNG > predict用画像
│   ├── The_Nighthawk_Star_op.PNG                > predict用画像
│   ├── test_32x32_000800.npz                    > 学習用データセット（サンプル）
│   └── train_32x32_007200.npz                   > 学習用データセット（サンプル）
├── LICENSE
├── Lib
│   ├── imgfunc.py  > 画像処理に関する便利機能
│   ├── network.py  > jpegcompのネットワーク部分
│   └── plot_report_log.py
├── README.md
├── Tools
│   ├── LICENSE
│   ├── README.md
│   ├── dot2png.py   > dot言語で記述されたファイルをPNG形式に変換する
│   ├── func.py      > 便利機能
│   ├── npz2jpg.py   > 作成したデータセット（.npz）の中身を画像として出力する
│   └── plot_diff.py > logファイルの複数比較
├── auto_train.sh
├── clean_all.sh
├── create_dataset.py        > 画像を読み込んでデータセットを作成する
├── predict.py               > モデルとモデルパラメータを利用して推論実行する
├── predict_some_snapshot.py > 複数のsnapshotoとひとつのモデルパラメータを利用してsnapshotの推移を可視化する
└── train.py                 > 学習メイン部
```

FontDataはチュートリアル用のデータセットとテスト用の画像しかない。完全版データは非常に重いので別リポジトリにて管理している。

# チュートリアル

## 1. 学習する

### 実行

```console
$ ./train.py -i FontData/
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
in_path:	FontData/
layer_num:	3
lossfun:	mse
only_check:	True
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
train_32x32_007200.npz:	comp(7200, 32, 32),	raw(7200, 32, 32)
test_32x32_000800.npz:	comp(800, 32, 32),	raw(800, 32, 32)
epoch       main/loss   validation/main/loss  elapsed_time
1           0.176155    0.153774              73.2628
2           0.146684    0.136503              147.235
3           0.134237    0.129488              221.06
4           0.124759    0.120349              294.067
5           0.117036    0.112505              366.737
6           0.110237    0.107868              440.326
7           0.104063    0.100743              513.166
8           0.0984077   0.0963099             587.07
9           0.0932294   0.0918062             660.406
10          0.0883972   0.0864184             733.576
```

`not import cupy`はcupyをインストールしていない場合に表示される

### 生成物の確認

resultフォルダ中に`*.json`、`*.log`、`*_graph.dot`、`*_log_plot.png`、`*.snapshot`、`*.model`が生成されていればOK

## 2. 学習で作成されたモデルを使用する

### 実行

```console
$  ./predict.py ./result/*.model ./result/*.json ./FontData/The_Night*
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
model:	./result/180126-140329.model
out_path:	./result/
param:	./result/180126-140329.json
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

resultフォルダ中に`comp-*.jpg`ファイルが生成されていればOK

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
$ ./Tools/dot2png.py ./result/*.dot
```

以下のような画像が生成される（例はMNISTのネットワーク層）。

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/cg.png" width="320px">


## NPZデータセットの中身をランダム表示

```console
$ ./Tools/npz2jpg.py ./FontData/test_32x32_000800.npz
```

以下のような画像が生成される。上段が圧縮画像、下段が無圧縮画像

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/npz2jpg.jpg" width="320px">


## データセットを作成する

### 構成

jpegcomp直下にいるものとする

```console
├── FontDataAll
│   ├── README.md
│   ├── The_Night_of_the_Milky_Way_Train_ch2.PNG
│   ├── The_Nighthawk_Star_op.PNG
│   ├── font_00.bmp
│   └── font_01.bmp
└── jpegcomp
     ├── FontData
     ├── Lib
     └── Tools
```

### 実行

```console
$ ./create_dataset.py ../FontDataAll/font_0[0-1]*
```

### 端末の確認
以下のとおりであれば正常に実行できている

```console
not import cupy
------------------------------
channel:	1
img_size:	32
jpeg[2]:
	../FontDataAll/font_00.bmp
	../FontDataAll/font_01.bmp
out_path:	./result/
quality:	5
round:	1000
train_per_all:	0.9
------------------------------
read images...
split images...
shuffle images...
train comp/raw:(7200, 32, 32)/(7200, 32, 32)
test  comp/raw:(800, 32, 32)/(800, 32, 32)
save npz...
```

### 生成物の確認

resultフォルダが作成され、その中に以下のファイルが生成されていればOK
- `test_32x32_000800.npz`
- `train_32x32_007200.npz`
