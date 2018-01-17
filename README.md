# 概要

圧縮されたJpeg画像の復元

# 動作環境

- OS
  - Ubuntu 16.04.3 LTS ($ cat /etc/issue)
- Python
  - 3.5.2 ($ python3 -V)
- chainer
  - 3.2 ($ pip3 show chainer | grep Ver)
- numpy
  1.13.3 ($ pip3 show numpy | grep Ver)

# ファイル構成

## 生成方法
o$ ls `find ./ -maxdepth 2 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt

## ファイル

- ./create_dataset.py:  '画像を読み込んでデータセットを作成する'
- ./dot2png.py:         'dot言語で記述されたファイルをPNG形式に変換する'
- ./func.py:            '便利機能'
- ./network.py:         'jpegcompのネットワーク部分'
- ./npz2jpg.py:         '画像を読み込んでデータセットを作成する'
- ./predict.py:         'スナップショットを利用した画像の生成'
- ./train.py:           '学習メイン部'

# チュートリアル

## データセットを作成する

### 実行

$ ./create_dataset.py ./FontData/font_0[0-1]*

### 端末の確認
以下のとおりであれば正常に実行できている

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

`not import cupy`はcupyをインストールしていない場合に表示される

### 生成物の確認

resultフォルダが作成され、その中にtest.npzとtrain.npzが生成されていればOK

## 学習する

### 実行

$ ./train.py

### 端末の確認

not import cupy
------------------------------
actfunc_1:	relu
actfunc_2:	sigmoid
batchsize:	100
epoch:	10
frequency:	-1
gpu_id:	-1
in_path:	./result/
layer_num:	3
lossfun:	mse
only_check:	False
out_path:	./result/
plot:	True
resume:
unit:	16
------------------------------
epoch       main/loss   validation/main/loss  elapsed_time
1           0.176222    0.183978              119.484
2           0.131534    0.124229              239.324
3           0.115318    0.115374              359.106
4           0.102909    0.100117              478.383
5           0.0924858   0.0898679             596.97
6           0.0835829   0.0809697             714.618
7           0.0757951   0.0713755             831.607
8           0.0689926   0.064955              950.95
9           0.0630572   0.0585681             1068.53
10          0.0578445   0.0542895             1185.45

### 生成物の確認

resultフォルダ中に*.log、*_graph.dot、*_plot.png、*.snapshot、*.modelが生成されていればOK

## 学習で作成されたモデルを使用する

### 実行

$ ./predict.py ./result/*.model ./FontData/The_Night*

### 端末の確認

not import cupy
------------------------------
batch:	100
channel:	1
gpu:	-1
img_size:	32
jpeg[2]:
	./FontData/The_Night_of_the_Milky_Way_Train_ch2.PNG
	./FontData/The_Nighthawk_Star_op.PNG
model:	./result/unit(16)_ch(1)_layer(3)_actFunc(relu_sigmoid)_180117-150852.model
out_path:	./result/
quality:	5
------------------------------

### 生成物の確認

resultフォルダ中にcomp-*.jpgファイルが生成されていればOK

# その他の機能

## 生成されたデータの削除

$ ./clean_all.sh

## ハイパーパラメータを変更して検証

$ ./auto_train.sh

## Dotファイルの画像化

$ ./dot2png.py ./result/*.dot