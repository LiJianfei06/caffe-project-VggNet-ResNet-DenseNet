#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=examples/ljftest_alphabet
DATA=examples/ljftest_alphabet
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/imagenet_mean.binaryproto

echo "Done."
