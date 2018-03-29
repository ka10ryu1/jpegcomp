Deep Learningで使用する便利なツール集

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
├── LICENSE
├── README.md
├── Tests
│   ├── Lenna.bmp       > テスト用画像
│   ├── Mandrill.bmp    > テスト用画像
│   ├── test_getfunc.py > getfuncのテスト用コード
│   └── test_imgfunc.py > imgfuncのテスト用コード'
├── dot2png.py        > dot言語で記述されたファイルをPNG形式に変換する
├── func.py           > 便利機能
├── getfunc.py        > 画像処理に関する便利機能
├── imgfunc.py        > 画像処理に関する便利機能
├── npz2jpg.py        > 作成したデータセット（.npz）の中身を画像として出力する
├── plot_diff.py      > logファイルの複数比較
└── png_monitoring.py > 任意のフォルダの監視
```

# 使い方

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
