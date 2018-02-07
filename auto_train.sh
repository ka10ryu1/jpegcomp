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

#echo "test1"
#./train.py -lf mse -a1 relu -a2 sigmoid -ln 4 -u 128 -b 200 -e 200 -g 0 -f 20 -o ./result/0121_test1/ $FLAG_CHK

echo "test1"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt sgd $FLAG_CHK

echo "test2"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt adam $FLAG_CHK

echo "test3"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt ada_d $FLAG_CHK

echo "test4"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt ada_g $FLAG_CHK

echo "test5"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt m_sgd $FLAG_CHK

echo "test6"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt n_ag $FLAG_CHK

echo "test7"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt rmsp $FLAG_CHK

echo "test8"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt rmsp_g $FLAG_CHK

echo "test9"
./train.py -i FontData/ -o ./result/011/ -ln 4 -b 10 -opt smorms $FLAG_CHK
