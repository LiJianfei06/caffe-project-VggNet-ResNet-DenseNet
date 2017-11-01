GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_alphabet_ResNet/Log/ ./build/tools/caffe train --solver=examples/ljftest_alphabet_ResNet/solver.prototxt 2>&1| tee examples/ljftest_alphabet_ResNet/caffe.log$@

