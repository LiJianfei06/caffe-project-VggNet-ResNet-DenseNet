GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_alphabet_VggNet_BN/Log/ ./build/tools/caffe train --solver=examples/ljftest_alphabet_VggNet_BN/solver.prototxt 2>&1| tee examples/ljftest_alphabet_VggNet_BN/caffe.log$@

