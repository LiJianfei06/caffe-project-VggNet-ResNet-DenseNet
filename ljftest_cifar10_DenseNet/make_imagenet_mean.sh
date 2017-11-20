#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/ljftest_cifar10_DenseNet
DATA=examples/ljftest_cifar10_DenseNet
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
