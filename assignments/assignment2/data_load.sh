#!/usr/bin/env bash

DIR=data

mkdir $DIR
kaggle competitions download -c comp-551-imbd-sentiment-classification -p $DIR

unzip -q $DIR/test.zip -d $DIR/
unzip -q $DIR/train.zip -d $DIR/
