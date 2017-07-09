#!/bin/bash

if [ $# -ne 2 ] ; then
    echo "./server_setup.sh <daynum> <usernum>"
    exit 1
fi

day=$1
numusers=$2
startport=8007
interval=10
numgpus=4

DATAPATH="/users/summerschool/cvdata"
CODEPATH="/users/summerschool/cvitcvss17"
WORKPATH="/tmp/summerschool"


echo "copying data.."
mkdir -p $WORKPATH/data
cp -r $DATAPATH/lab${day} $WORKPATH/data/

for i in `seq 1 $numusers` ; do
    destdir="$WORKPATH/user${i}"
    mkdir -p $destdir
    cp -av $CODEPATH/lab${day} $destdir
    cd $destdir
    #echo `pwd`
    #echo $(($startport+($i-1)*$interval))
     echo $(($i%$numgpus))
     CUDA_VISIBLE_DEVICES=$(($i%$numgpus)) jupyter notebook --no-browser --ip="0.0.0.0" --port=$(($startport+($i-1)*$interval)) &
    cd
done
