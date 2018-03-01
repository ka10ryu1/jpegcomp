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
actfun_2:	sigmoid
batchsize:	100
dropout:	0.1
epoch:	10
frequency:	-1
gpu_id:	-1
in_path:	FontData/
layer_num:	2
lossfun:	mse
network:	0
only_check:	False
optimizer:	adam
out_path:	./result/
plot:	True
resume:
shuffle_rate:	2
unit:	2
------------------------------
Activation func: relu
Activation func: sigmoid
[Network info] JC_DDUU
  Unit:	2
  Out:	1
  Layer:	2
  Drop out:	0.1
Act Func:	relu, sigmoid
Loss func: mean_squared_error
Optimizer: Adam optimizer
train_32x32_010800.npz:	x(10800, 1, 32, 32),	y(10800, 1, 32, 32)
test_32x32_001200.npz:	x(1200, 1, 32, 32),	y(1200, 1, 32, 32)
epoch       main/loss   validation/main/loss  lr          elapsed_time
1           0.201806    0.175782              0.000320036  22.9336
2           0.157518    0.145553              0.000440853  47.8616
3           0.136482    0.127626              0.000526182  72.3183
4           0.119958    0.112415              0.000592394  97.2897
5           0.10619     0.0987265             0.000646072  122.325
6           0.0944588   0.0891305             0.000690709  146.864
7           0.0841792   0.079238              0.000728448  171.068
8           0.0751524   0.0699162             0.000760729  195.549
9           0.0669822   0.0689321             0.00078858  220.218
10          0.0571909   0.0512061             0.000812766  244.215
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
jpeg[2]:
	FontData/The_Night_of_the_Milky_Way_Train_ch2.PNG
	FontData/The_Nighthawk_Star_op.PNG
model:	result/fff0ljg.model
out_path:	./result/
param:	result/fff0ljg.json
quality:	5
------------------------------
model param: result/fff0ljg.json
Activation func: relu
Activation func: sigmoid
[Network info] JC_DDUU
  Unit:	2
  Out:	1
  Layer:	2
  Drop out:	0.0
Act Func:	relu, sigmoid
model read: result/fff0ljg.model
exec time: 0.48[s]
save: ./result/comp-001.jpg
exec time: 0.34[s]
```

### 生成物の確認

resultフォルダ中に`comp-*.jpg`と`concat-*.jpg`が生成されていればOK。



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

jpegcompにあるデータセットは最小構成であるため、完全版データを作成するためにはd[FontDataAllリポジトリ](https://github.com/ka10ryu1/FontDataAll)を持ってくる必要がある。
以下ではFontDataAllとjpegcompを同じフォルダー内にcloneし、jpegcomp直下にいるものとする。

```console
├── FontDataAll
│   ├── README.md
│   ├── The_Night_of_the_Milky_Way_Train_ch2.PNG
│   ├── The_Nighthawk_Star_op.PNG
│   ├── font_00.bmp
│   ├── font_01.bmp
│   ├──
│   ├──
│   └── font_3f.bmp
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

### 画像の比較

`concat_3images.py`を利用することで、`predict.py`と同様の`concat-*.jpg`を生成できる。推論実行したくない時やその環境でない時に使用する。以下のように実行する。

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
