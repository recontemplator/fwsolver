#!/bin/bash
#sort  dict/dict.out|uniq>dict/dict2.txt

gzip -dc dict/dict.txt.gz |grep -Fvxf dict/stop_words.txt >dict/dict_filtered.txt

rm letters.dat
./solver.py img/train1.png аропивделсямбхунцк
./solver.py img/train2.png тзшфчь
./solver.py img/train3.png гжйёы
./solver.py img/train4.png щ
./solver.py img/train5.png ю
./solver.py img/train6.png э

