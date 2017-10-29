GLOG_logtostderr=0 GLOG_log_dir=examples/ljftest_alphabet_VggNet/Log/ ./build/tools/caffe train --solver=examples/ljftest_alphabet_VggNet/solver.prototxt 2>&1| tee examples/ljftest_alphabet_VggNet/caffe.log$@

