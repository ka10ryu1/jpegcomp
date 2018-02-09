#!/bin/bash
# auto_train.sh
# train.pyをいろいろな条件で試したい時のスクリプト
# train.pyの引数を手入力するため、ミスが発生しやすい。
# auto_train.shを修正したら、一度-cオプションを実行してミスがないか確認するべき

# オプション引数を判定する部分（変更しない）

usage_exit() {
    echo "Usage: $0 [-c]" 1>&2
    echo " -c: 設定が正常に動作するか確認する"
    exit 1
}

FLAG_CHK=""
while getopts ch OPT
do
    case $OPT in
        c)  FLAG_CHK="--only_check"
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done

shift $((OPTIND - 1))

# 以下自由に変更する部分（オプション引数を反映させるなら、$FLG_CHKは必要）

COUNT=1
echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -i FontData/ -o ./result/001/ -b 10 -e 50 $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -i FontData/ -o ./result/001/ -b 20 -e 50 $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -i FontData/ -o ./result/001/ -b 30 -e 50 $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -i FontData/ -o ./result/001/ -b 50 -e 50 $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -i FontData/ -o ./result/001/ -b 70 -e 50 $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo -e "\n<< test ["${COUNT}"] >>\n"
./train.py -i FontData/ -o ./result/001/ -b 100 -e 50 $FLAG_CHK
COUNT=$(( COUNT + 1 ))
