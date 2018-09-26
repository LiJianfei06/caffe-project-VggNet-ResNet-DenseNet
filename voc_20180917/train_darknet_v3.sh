#!/usr/bin/env sh

CAFFE_HOME=../../..

SOLVER=./solver.prototxt
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_multifixed_iter_2000.solverstate
SNAPSHOT=./models_save/yolov2_VOC0712_iter_38000.solverstate
#WEIGHTS=../convert/yolo.caffemodel
#WEIGHTS=./models_save/yolov2_VOC0712_iter_20000.caffemodel
#SNAPSHOT=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_120000.solverstate

#$CAFFE_HOME/build/tools/caffe train \
#    --solver=$SOLVER --weights=$WEIGHTS \
#    2>&1 | tee train_darknet_anchor1.log 
#--weights=$WEIGHTS #--gpu=0,1


$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --snapshot=$SNAPSHOT \
    2>&1 | tee train_darknet_anchor.log #--weights=$WEIGHTS #--gpu=0,1
