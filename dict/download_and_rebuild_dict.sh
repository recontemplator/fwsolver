#!/bin/bash

wget http://opencorpora.org/files/export/dict/dict.opcorpora.xml.bz2
bzip2 -dkf dict.opcorpora.xml.bz2
./generate_dict.py
sort dict.out|uniq>dict.txt