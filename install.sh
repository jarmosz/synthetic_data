#!/bin/bash
apt update -y
apt upgrade -y

apt-get install python3 python3-pip python3-venv unzip gawk screen python-dev libhunspell-dev
python3 -m venv env
source ./env/bin/activate
pip install pandas spacy regex scipy hunspell
python -m spacy download pl_core_news_lg

mkdir data && cd data
wget https://minio.clarin-pl.eu/ermlab/public/PoLitBert/corpus-oscar/corpus_oscar_2020-04-10_64M_lines.zip
unzip -p corpus_oscar_2020-04-10_64M_lines.zip | sed -r '/^\s*$/d' | gawk 'NF>6'  > oscar_filtered.txt
split -l 1000000 --numeric-suffixes=1 --suffix-length=1 --additional-suffix=".txt"  oscar_filtered.txt ""