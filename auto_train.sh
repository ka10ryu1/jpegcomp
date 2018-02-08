#!/bin/bash
# auto_train.sh
# train.pyをいろいろな条件で試したい時のスクリプト
# train.pyの引数を手入力するため、ミスが発生しやすい。
# auto_train.shを修正したら、一度-cオプションを実行してミスがないか確認するべき

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

COUNT=1
echo "test"${COUNT}
./train.py -i FontData/ -o ./result/002/ -ln 4 -b 10 -e 30 -opt smorms $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo "test"${COUNT}
./train.py -i FontData/ -o ./result/002/ -ln 4 -b 20 -e 30 -opt smorms $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo "test"${COUNT}
./train.py -i FontData/ -o ./result/002/ -ln 4 -b 30 -e 30 -opt smorms $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo "test"${COUNT}
./train.py -i FontData/ -o ./result/002/ -ln 4 -b 50 -e 30 -opt smorms $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo "test"${COUNT}
./train.py -i FontData/ -o ./result/002/ -ln 4 -b 80 -e 30 -opt smorms $FLAG_CHK
COUNT=$(( COUNT + 1 ))

echo "test"${COUNT}
./train.py -i FontData/ -o ./result/002/ -ln 4 -b 100 -e 30 -opt smorms $FLAG_CHK
COUNT=$(( COUNT + 1 ))
