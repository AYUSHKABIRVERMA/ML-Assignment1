#!/bin/sh

if [ "$1" -eq "1" ]
then
	python3 LRa.py $2 $3 $4 $5
elif [ "$1" -eq "2" ]
then
	python3 WLRb.py $2 $3 
elif [ "$1" -eq "3" ]
then
	python3 LoR.py $2 $3
elif [ "$1" -eq "4" ]
then
	python3 GDA.py $2 $3 $4
else
	echo "Enter a number between 1 to 4"
fi