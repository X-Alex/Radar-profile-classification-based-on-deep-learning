# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:44:20 2017

@author: zfh
"""
import os
import re
from skimage import io,transform
import tensorflow as tf
import numpy as np
import time
import geopandas as gp

from matplotlib import pyplot as plt
from skimage import draw
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import utils
import mapmark
#迭代计数器
record_count = 0
#睡眠时间
time_kick = 3
#压缩图片尺寸
w=100
h=100
c=3
#网格宽和高
gridH = 50
gridC = 50
#bin文件基本信息
startLon = 115.00
startLat = 34.00
endLon = 119.55
endLat = 29.45
radarCount = 5
XReso = 9.00
YReso = 9.00
#站点名称
station = "Z0551"

#批量处理个数
batch = 200
#气象分类
mete_digcategory = {0:'0',1:'1',2:'2',3:'3'}
mete_category = ['norain-nowind','norain-wind','rain-nowind','rain-wind']

root_path = "C:/Users/zfh/Desktop/newpic"
target_obj = "cappiprofilelist.txt"

#模型路径
model_path = "C:/Users/zfh/Desktop/machineLearnDemo/RadarImage_recosys/radar_net2"
model_ckpt = "C:/Users/zfh/Desktop/machineLearnDemo/RadarImage_recosys/radar_net2/model.ckpt.meta"
#list清单绝对路径
ab_rootPath = "C:/Users/zfh/Desktop/newpic/cappiprofilelist.txt"

#在读取地图时，地图角标色彩
map_tagger = [(150,0,180),(0,160,248),(255,0,0),(255,255,255)]
#shp文件的路径
shpfile_path = 'C:/Users/zfh/Desktop/anhui/areaClip.shp'
#设置地图上的字体
font = ImageFont.truetype("C:\Windows\Fonts\Arial.ttf", 30)
#生成地图路径
map_path = "C:/Users/zfh/Desktop/machineLearnDemo/RadarImage_recosys/target.jpg"

def obtain_data(root_path,target_obj,path,data,ab_rootPath):
    test_objectFile = os.listdir(root_path)
    time.sleep(time_kick)
    if target_obj in test_objectFile:
        f = open(ab_rootPath)
        while 1:
            line = f.readline()
            if not line:
                break
            path.append(line.rstrip('\n'))
        f.close()
        if len(path)-1 == 1:
            filetime,mid_path,generateTime,title = utils.empty_grid(crsave,result_output,path,gridH,gridC,station)
            utils.create_bin(crsave,result_output,title,filetime,generateTime,station,gridH,gridC,radarCount,startLon,startLat,endLon,endLat,XReso,YReso)
            os.remove(ab_rootPath)
            return 0,0,0
        else:
            for obj_path in path[1:]:
                rel_path = obj_path.split(" ")[2]
                l_data = read_one_image(rel_path)
                data.append(l_data)#封装数据
                i,j = get_cr(rel_path)
                obj = pack_cr(int(i),int(j))
                crsave.append(obj)#封装数据所对应的坐标
            return path,data,crsave
    else:
        return 0,0,0
#通过正则表达式获得行号和列号
def get_cr(path):
    p1 = "R\d+C\d+"
    obj = re.compile(p1).findall(path)[0]
    cr = re.findall("\d+",obj)
    return cr[0],cr[1]
#封装成列表[i,j]
def pack_cr(i,j):
    packcr = []
    packcr.append(i)
    packcr.append(j)
    return packcr
#读取单个图片
def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

"""
def selectResult(result_out)通过选票机制从两种角度中选择权限最高的识别结果
result_out:识别结果集合
"""
def selectResult(result_out,crsave):
    select_result = []
    select_crsave = []
    for i in range(len(result_out)):
        if i%2 == 0:
            select_result.append(max(result_out[i],result_out[i+1]))
            select_crsave.append(crsave[i])
    return select_result,select_crsave

#识别图像
#整体识别
def whoreco_result(data,n):
    feed_dict = {x:data}
    output = []
    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits,feed_dict)
    output = tf.argmax(classification_result,1).eval()
    #识别
    for i in range(batch):
        n = n+1
        path[n] = path[n].split(" ")[2]
        print(path[n]+'的类型是:'+mete_digcategory[output[i]])
        result_output.append(mete_digcategory[output[i]])
#部分识别
def partreco_result(data,n):
    feed_dict = {x:data}
    output = []
    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits,feed_dict)
    output = tf.argmax(classification_result,1).eval()
    #识别
    for i in range(len(data)):
        n = n+1
        path[n] = path[n].split(" ")[2]
        print(path[n]+'的类型是:'+mete_digcategory[output[i]])
        result_output.append(mete_digcategory[output[i]])

#调用方法生成地图
mapmark.generate_map(shpfile_path,"anhuiMap.png")
#调用方法标记地图
mapmark.mark_map("anhuiMap.png",mete_category,map_tagger,font)

#自动化运行
while True:
    #存储识别出来的结果集合[0,1,2,1,...,3...]
    result_output = []
    #data列表负责存储由read_one_image返回的矩阵数据
    data = []
    #path列表负责存储路径[ProfileProduct SND D:/Web/2018-04-03/x.png\n]
    path = []
    #保存遍历根目录文件项
    test_objectFile = []
    #存储行列的列表
    crsave = []
    try:
        path,data,crsave = obtain_data(root_path,target_obj,path,data,ab_rootPath)
        
        if path==0 and data==0:
            continue
        else:
            print(len(data))
            with tf.Session() as sess:
                print("open it")
                saver = tf.train.import_meta_graph(model_ckpt)
                print("inter model")
                saver.restore(sess,tf.train.latest_checkpoint(model_path))
                graph = tf.get_default_graph()
                x = graph.get_tensor_by_name("x:0")
                if len(data) % batch == 0:
                    for i in range(len(data) // batch):
                        minibatch = data[i*batch:(i+1)*batch]
                        whoreco_result(minibatch,i*batch)
                elif len(data) < batch:
                    minibatch = data[0:len(data)]
                    partreco_result(minibatch,0)
                else:
                    whoRange = len(data) // batch
                    partRange = len(data) % batch
                    for i in range(whoRange):
                        minibatch = data[i*batch:(i+1)*batch]
                        whoreco_result(minibatch,i*batch)
                    minibatch = data[whoRange*batch:len(data)]
                    partreco_result(minibatch,whoRange*batch)
                result_output,crsave = selectResult(result_output,crsave)
                os.remove(ab_rootPath)
                filetime,generateTime,title = utils.create_grid(crsave,result_output,path,gridH,gridC,station)
                utils.create_bin(crsave,result_output,title,filetime,generateTime,station,gridH,gridC,radarCount,startLon,startLat,endLon,endLat,XReso,YReso)
    except:
        os.remove(ab_rootPath)
        pass
    record_count += 1
    print(record_count)