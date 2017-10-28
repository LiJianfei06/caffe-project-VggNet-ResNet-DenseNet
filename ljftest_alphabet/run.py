# -*- coding: UTF-8 -*-
import sys
                                       

sys.path.append("/home/ljf/caffe-master/python")
sys.path.append("/home/ljf/caffe-master/python/caffe")
import caffe     
from caffe import layers as L,params as P,to_proto
import tools








root_str="/home/ljf/caffe-master/examples/ljftest_alphabet/"
solver_dir = root_str + 'solver.prototxt'



if __name__ == '__main__':
   #下面的代码和上面的代码要分开执行，除非 solver 的 prototxt 不需要修改
    caffe.set_device(0) #设置使用的 GPU 编号
    caffe.set_mode_gpu()#设置迭代采用 GPU 加速模式
    solver = caffe.SGDSolver(str(solver_dir))
    for _ in range(650):
        solver.step(100) 