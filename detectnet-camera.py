# -*- coding:utf-8 -*-
# 用于识别车辆(基于KITTI数据集)
import os
os.environ['GLOG_minloglevel'] = '2' # 将caffe的输出log信息不显示，必须放到import caffe前
import caffe # caffe 模块
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 识别图片中的车辆
def detection(net, transformer):
    
    #读取图片、执行已经设置好的图片处理过程并导入学习网络中
    im = caffe.io.load_image('car.png')
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)  #将BGR转成RGB
    #识别并计算消耗时间
    start = time.clock()
    net.forward()# 用训练好的网络识别
    end = time.clock()
    print('detection time: %f s' % (end - start))

    # 获取目标检测结果	
    loc = net.blobs['bbox-list'].data[0]
    #绘画目标方框
    #查看了结构文件发现在CAFFE一开始图像输入的时候就已经将图片缩小了，宽度1248高度384
    #然后我们在net.blobs['bbox-list'].data得到的是侦测到的目标座标，但是是相对于1248*384的
    #所以我们要把座标转换回相对原大小的位置，下面im.shape是保存在原尺寸的宽高，
    for l in range(len(loc)):
		xmin = int(loc[l][0] * im.shape[1] / 1248)
		ymin = int(loc[l][1] * im.shape[0] / 384)
		xmax = int(loc[l][2] * im.shape[1] /1248)
		ymax = int(loc[l][3] * im.shape[0] / 384)
		#在该座标位置画一个方框
		cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (55 / 255.0, 255 / 255.0, 155 / 255.0), 2)
    # 显示结果
    return im


#CPU或GPU模型转换
#caffe.set_mode_cpu()
#caffe.set_device(0)
caffe.set_mode_gpu()

#caffe根目录
caffe_root = '/home/nvidia/caffe/'
# 网络参数（权重）文件
caffemodel =  'detecnet.caffemodel'
# 网络实施结构配置文件
deploy = 'deploy.prototxt'


# 网络实施分类
net = caffe.Net(deploy,  # 定义模型结构
                caffemodel,  # 包含了模型的训练权值
                caffe.TEST)  # 使用测试模式(不执行dropout)

# 加载ImageNet图像均值 (随着Caffe一起发布的)
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # 对所有像素值取平均以此获取BGR的均值像素值

# 图像预处理
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) 
transformer.set_transpose('data', (2,0,1))  
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255)

#计数标识
count = 0
#打开摄像头
cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)I420, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

# 显示检测视频
while True:
    count  = count  + 1
    ret,frame = cap.read()#从摄像头获取图片
    #每十帧检测一次目标
    if count ==10 :
        cv2.imwrite("car.png", frame)#存储成图片
        frame = detection(net,transformer)
        count  = 0
    cv2.imshow("capture", frame)#显示
    if cv2.waitKey(100) & 0xFF == ord('q'):#按q结束
        break
