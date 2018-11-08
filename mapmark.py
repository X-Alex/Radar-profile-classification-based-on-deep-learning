# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 14:21:30 2017

@author: zfh
"""
import os

from skimage import io,transform
import tensorflow as tf
import numpy as np
import geopandas as gp

from matplotlib import pyplot as plt
from skimage import draw
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

"""
def generate_map(url,save_path)读取shp文件并另存为指定格式
url:路径
save_path:指定另存为格式
"""
def generate_map(url,save_path):
    anhui_pro = gp.GeoDataFrame.from_file(url)
    anhui_pro.head()
    anhui_pro.plot(color = '#FFFFFF',edgecolor = 'black',linewidth = 1.0)
    plt.savefig(save_path,dpi = 300)
    plt.show()
    plt.close()
"""
def mark_map(url,wearth_col,color,font)在生成地图的基础上对地图进行颜色标注
url:生成地图的路径
wearth_col:天气种类说明 mete_category
color:天气种类所对应的颜色 map_tagger
font:字体类型
"""
def mark_map(url,wearth_col,color,font):
    img = Image.open(url)
    draw = ImageDraw.Draw(img)
    x,y=1400,800
    
    rec_up,rec_down = 810,830
    for cls,col in zip(wearth_col,color):
        draw.rectangle([1370,rec_up,1390,rec_down],fill = col)
        draw.text((x,y),cls,(0,0,0),font = font)
        draw = ImageDraw.Draw(img)
        y += 50
        rec_up += 50
        rec_down += 50
    img.save("target2.png")
