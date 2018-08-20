#!/usr/bin/env sh

CAFFE_HOME=../..

SOLVER=./solver.prototxt
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_multifixed_iter_2000.solverstate
#WEIGHTS=./yolo.caffemodel
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_120000.solverstate
$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=0 \
    2>&1 | tee train_ResNet-56.log #--weights=$WEIGHTS #--gpu=0,1

