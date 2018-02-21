# 概要

圧縮されたJpeg画像の復元

## 学習結果（5%に圧縮した画像を復元）

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-final-00.jpg" width="640px">

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-final-10.jpg" width="640px">

- **Original** : 圧縮なしの画像
- **Compression** : Originalを圧縮した画像
- **Restoration** : Compressionを学習によって復元した画像

# 動作環境

- **Ubuntu** 16.04.3 LTS ($ cat /etc/issue)
- **Python** 3.5.2 ($ python3 -V)
- **chainer** 3.2 ($ pip3 show chainer | grep Ver)
- **numpy** 1.13.3 ($ pip3 show numpy | grep Ver)
- **cupy** 2.2 ($ pip3 show cupy | grep Ver)
- **opencv-python** 3.4.0.12 ($ pip3 show opencv-python | grep Ver)

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 3 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
$ tree >& log.txt
```

## ファイル

```console
.
├── FontData
│   ├── The_Night_of_the_Milky_Way_Train_ch2.PNG > predict用画像
│   ├── The_Nighthawk_Star_op.PNG                > predict用画像
│   ├── test_32x32_000800.npz                    > 学習用データセット（サンプル）
│   └── train_32x32_007200.npz                   > 学習用データセット（サンプル）
├── LICENSE
├── Lib
│   ├── Tests
│   │   ├── Lenna.bmp       > テスト用画像
│   │   ├── Mandrill.bmp    > テスト用画像
│   │   └── test_imgfunc.py > imgfuncのテスト用コード
│   ├── imgfunc.py  > 画像処理に関する便利機能
│   ├── network.py  > jpegcompのネットワーク部分
│   ├── network2.py > jpegcompのネットワーク部分その2
│   └── plot_report_log.py
├── README.md
├── Tools
│   ├── LICENSE
│   ├── README.md
│   ├── dot2png.py        > dot言語で記述されたファイルをPNG形式に変換する
│   ├── func.py           > 便利機能
│   ├── npz2jpg.py        > 作成したデータセット（.npz）の中身を画像として出力する
│   ├── plot_diff.py      > logファイルの複数比較
│   └── png_monitoring.py > 任意のフォルダの監視
├── auto_train.sh
├── clean_all.sh
├── concat_3_images.py       > 3枚の画像を連結する（org, comp, restration）
├── create_dataset.py        > 画像を読み込んでデータセットを作成する
├── predict.py               > モデルとモデルパラメータを利用して推論実行する
├── predict_some_snapshot.py > 複数のsnapshotoとひとつのモデルパラメータを利用してsnapshotの推移を可視化する
└── train.py                 > 学習メイン部
```

FontDataはチュートリアル用のデータセットとテスト用の画像しかない。完全版データは非常に重いので[別リポジトリ](https://github.com/ka10ryu1/FontDataAll)にて管理している。

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
actfun_2:	h_sigmoid
batchsize:	100
dropout:	0.1
epoch:	10
frequency:	-1
gpu_id:	-1
in_path:	FontData/
layer_num:	2
lossfun:	mse
only_check:	False
optimizer:	adam
out_path:	./result/
plot:	True
resume:
shuffle_rate:	2
unit:	4
------------------------------
Activation func: relu
Activation func: hard_sigmoid
[Network info]
  Unit:	4
  Out:	1
  Layer:	2
  Drop out:	0.1
Act Func:	relu, hard_sigmoid
Loss func: mean_squared_error
Optimizer: Adam optimizer
train_32x32_010800.npz:	x(10800, 32, 32),	y(10800, 32, 32)
test_32x32_001200.npz:	x(1200, 32, 32),	y(1200, 32, 32)
epoch       main/loss   validation/main/loss  lr          elapsed_time
1           0.180935    0.145807              0.000320036  63.9726
2           0.147242    0.130009              0.000440853  128.258
3           0.127155    0.114856              0.000526182  191.622
4           0.108269    0.0965278             0.000592394  255.361
5           0.0925587   0.0828782             0.000646072  319.02
6           0.0798688   0.0721286             0.000690709  382.199
7           0.0691529   0.0619116             0.000728448  444.277
8           0.0598662   0.0525341             0.000760729  508.985
9           0.0518347   0.0449867             0.00078858  573.866
10          0.0448583   0.0391221             0.000812766  637.006
```

`not import cupy`はcupyをインストールしていない場合に表示される

### 生成物の確認

resultフォルダ中に以下が生成されていればOK。先頭の文字列は日付と時間から算出された値である。
- `*.json`
- `*.log`
- `*.model`
- `*_10.snapshot`
- `*_graph.dot`
- `loss.png`
- `lr.png`

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
model:	./result/iuj7l3s.model
out_path:	./result/
param:	./result/iuj7l3s.json
quality:	5
------------------------------
model param: ./result/iuj7l3s.json
Activation func: relu
Activation func: hard_sigmoid
[Network info]
  Unit:	4
  Out:	1
  Layer:	2
  Drop out:	0.0
Act Func:	relu, hard_sigmoid
model read: ./result/iuj7l3s.model
exec time: 1.10[s]
save: ./result/comp-001.jpg
exec time: 0.78[s]
save: ./result/comp-011.jpg
```

### 生成物の確認

resultフォルダ中に`comp-*.jpg`ファイルが生成されていればOK。

### 画像の比較

`concat_3images.py`を利用することで、オリジナル画像と圧縮画像と生成画像の比較が簡単にできる。以下のように実行する。

```console
$ ./concat_3_images.py ./FontData/The_Night_of_the_Milky_Way_Train_ch2.PNG ./result/comp-00*
```

以下のような画像が生成される。

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-00.jpg" width="320px">

```console
$ ./concat_3_images.py ./FontData/The_Nighthawk_Star_op.PNG ./result/comp-01*
```

以下のような画像が生成される。

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/concat-01.jpg" width="320px">


# その他の機能

## 生成されたデータの削除

```console
$ ./clean_all.sh
```

## ハイパーパラメータを変更して自動実行

デフォルトではバッチサイズだけを変更した学習を複数回繰り返す。

```console
$ ./auto_train.sh
```

## Dotファイルの画像化

学習を実行すると`*graph.dot`というファイルが生成されるので、それを画像化する。

```console
$ ./Tools/dot2png.py ./result/*.dot
```

以下のような画像が生成される（例はMNISTのネットワーク層）。

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/cg.png" width="320px">


## NPZデータセットの中身をランダム表示

学習用の画像を確認する。

```console
$ ./Tools/npz2jpg.py ./FontData/test_32x32_000800.npz
```

以下のような画像が生成される。上段が圧縮画像、下段が無圧縮画像

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/npz2jpg.jpg" width="320px">

## lossやlrの比較をする

### 学習を複数回実行する

`auto_train.sh`等で複数回学習させておく。以下はデフォルトの`auto_train.sh`を実行した前提である。

### 可視化を実行

```console
$ ./Tools/plot_diff.py ./result/001 -l all
```

以下のような画像が生成される。

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/plot_diff_loss.png" width="320px">

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/plot_diff_lr.png" width="320px">

※ファイルの作成日時順でソートされる。

## フォルダの監視

学習が更新されたら（loss.pngとlr.pngが更新されたら）Dropboxにコピーする場合

### 監視の実行

```console
$ ./Tools/png_monitoring.py ./result/ ~/Dropbox/temp/
```

以下が表示される。`Ctrl-c`連打すると監視を終了する。

```console
Monitoring : ./result/
Copy to : /home/aaaa/Dropbox/temp/
Exit: Ctrl-c
```

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
$ ./create_dataset.py ../FontDataAll/font_* -s 128 -r 2000 -t 0.95
```

### 生成物の確認

resultフォルダが作成され、その中に以下のファイルが生成されていればOK
- `test_128x128_******.npz`
- `train_128x128_******.npz`

※READMEの最上部にある結果は、このデータを使用している。

## スナップショットの進捗具合を可視化する

各スナップショットで推論実行したものを比較することで学習がどれだけ進んでいるかを可視化する。

### スナップショットの保存

1エポックずつスナップショットを保存する設定で学習を実行する。

```console
$ ./train.py -i FontData/ -f 1
```

### スナップショットの可視化

```console
$ ./predict_some_snapshot.py ./result/ ./FontData/The_Night_of_the_Milky_Way_Train_ch2.PNG -r 3 -n 8 -rs 8
```

### 生成物

以下のような画像が生成される。一番左が正解画像で、右に行くほど新しいスナップショットの結果になる。

<img src="https://github.com/ka10ryu1/jpegcomp/blob/images/Image/snapshots.jpg" width="320px">
