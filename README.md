Deep Learningで使用する便利なツール集

# 動作環境

- **Ubuntu** 16.04.4 LTS ($ cat /etc/issue)
- **Python** 3.5.2 ($ python3 -V)
- **chainer** 4.0.0 ($ pip3 show chainer | grep Ver)
- **numpy** 1.14.2 ($ pip3 show numpy | grep Ver)
- **cupy** 4.0.0 ($ pip3 show cupy | grep Ver)
- **opencv-python** 3.4.0.12 ($ pip3 show opencv-python | grep Ver)

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 3 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
$ tree >& log.txt
```

## ファイル

```console
├── LICENSE
├── README.md
├── Tests
│   ├── Lenna.bmp       > テスト用画像
│   ├── Mandrill.bmp    > テスト用画像
│   ├── test_getfunc.py > getfuncのテスト用コード
│   └── test_imgfunc.py > imgfuncのテスト用コード'
├── concat.py         > 複数の画像を任意の行列で結合する
├── dot2png.py        > dot言語で記述されたファイルをPNG形式に変換する
├── func.py           > 便利機能
├── getfunc.py        > 各種パラメータ取得に関する便利機能
├── imgfunc.py        > 画像処理に関する便利機能
├── npz2jpg.py        > 作成したデータセット（.npz）の中身を画像として出力する
├── plot_diff.py      > logファイルの複数比較
├── png_monitoring.py > 任意のフォルダの監視
└── pruning.py        > モデルの枝刈をする
```

# 使い方

## 複数の画像を任意の行列で結合する

以下のように入力すると、resultフォルダに4枚の画像が連結された画像が保存される。

```console
$ ./concat.py Tests/Lenna.bmp Tests/Mandrill.bmp Tests/Lenna.bmp Tests/Mandrill.bmp -r 2
```


## dot言語で記述されたファイルをPNG形式に変換する

以下のように入力すると、resultフォルダに`[dot言語で記述されたファイル]`が画像化されて保存される。形式はデフォルトでPNG、他にSVG、PDFに対応。

```console
$ ./dot2png.py [dot言語で記述されたファイル]
```

## 作成したデータセット（.npz）の中身を画像として出力する

以下のように入力すると、resultフォルダに`[データセット]`から抽出したデータが画像化されて保存される。抽出する画像は乱数を変更することで変更される。上段が入力画像、下段が正解画像。

```console
$ ./npz2png.py [データセット]
```

## logファイルの複数比較

以下のように入力すると、`[logファイルのあるフォルダ] ...`に格納された`.log`ファイルを探索してプロットし、resultフォルダに画像化して保存される。フォルダは複数選択できる。

```console
$ ./plot_diff.py [logファイルのあるフォルダ] ...
```

## 任意のフォルダの監視

以下のように入力すると、`[監視するフォルダ]`に新しい`.png`ファイルが格納された時に`[コピー先フォルダ]`にコピーする。`[コピー先フォルダ]`をDropboxなどのクラウドサービスに指定しておけば外出先でも進捗具合を確認できる。

```console
$ ./png_monitoring.py [監視するフォルダ] [コピー先フォルダ]
```

## モデルの枝刈をする

これ単体では動作しない。学習にスナップショットを利用する際にこれを使うと結果に寄与しにくいパラメータを削除してモデルの容量を削減できる。詳細は[リンク](http://tosaka2.hatenablog.com/entry/2017/11/17/194051)を参照されたい。

## テストの実行

`imgfunc.py`と`getfunc.py`はコードが肥大化しているため、テスト実行環境を用意して適宜確認している。
テストには`unittest`を利用している。
以下のように実行することでテストを実行できる。

```console
$ python -m unittest Tests/test_imgfunc.py
$ python -m unittest Tests/test_getfunc.py
```

実行後に以下のような表示がされればテストOK

```console
:
（省略）
:
----------------------------------------------------------------------
Ran 10 tests in 0.063s

OK
```
