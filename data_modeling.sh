#!/bin/sh

TRAIN_DATA="RARI_training.txt"
TEST_DATA="RARI_testing.txt"
NEW_DATA="./new_datas/*"

if [ -e $TRAIN_DATA ]
then
	rm $TRAIN_DATA
fi
if [ -e $TEST_DATA ]
then
	rm $TEST_DATA
fi
if [ -e $NEW_DATA ]
then
	mv $NEW_DATA "new_datas.txt"
fi

echo "Convert Data into Training, Testing format!"
python3 data_convert.py

echo "Training Data!"
python3 RARI.py | grep "MODEL" | cut -d " " -f 2-6 | python3 RARI_test.py
#python3 RARI.py
#python3 RARI_testing.py
