# -*- coding: utf-8 -*-
import struct
import os
import numpy as np
from skimage import draw
from skimage import io,transform

map_path = "C:/Users/zfh/Desktop/machineLearnDemo/RadarImage_recosys/target.jpg"
"""
def str2Bytes(a)将str类型转化为字节类型
def int2Bytes(c)将int类型转化为字节类型
"""
def str2Bytes(a):
    b = bytes(a,encoding='utf-8')
    return struct.pack(str(len(a))+'s',b)
def int2Bytes(c):
    b = int(c)
    return b

#封装bin文件
def create_bin(crsave,result_output,title,filetime,generateTime,station,gridH,gridC,radarCount,startLon,startLat,endLon,endLat,XReso,YReso):
    dataName = str(filetime[0:4])+'-'+str(filetime[4:6])+'-'+str(filetime[6:8])+' '+str(filetime[8:10])+':'+str(filetime[10:12])+':'+'00'+' '+'Finding Storm'
    DataName =  str2Bytes('DataName=%s,'%dataName)
    year = int(filetime[0:4])
    Year = str2Bytes('Year=%i,'%year)
    month = int(filetime[4:6])
    Month = str2Bytes('Month=%i,'%month)
    day = int(filetime[6:8])
    Day = str2Bytes('Day=%i,'%day)
    hour = int(filetime[8:10])   
    Hour = str2Bytes('Hour=%i,'%hour)
    minute = int(filetime[10:12])
    Minute = str2Bytes('Minute=%i,'%minute)
    
    xNumGrids = gridH
    yNumGrids = gridC
    XNumGrids = str2Bytes('XNumGrids=%i,'%xNumGrids)
    YNumGrids = str2Bytes('YNumGrids=%i,'%yNumGrids)
    RadarCount = str2Bytes('RadarCount=%i,'%radarCount)
    
    StartLon = str2Bytes('StartLon=%f,'%startLon)
    StartLat = str2Bytes('StartLat=%f,'%startLat)
    EndLon = str2Bytes('EndLon=%f,'%endLon)
    EndLat = str2Bytes('EndLat=%f,'%endLat)
    XReso = str2Bytes('XReso=%f,'%XReso)
    YReso = str2Bytes('YReso=%f,'%YReso)
    
    RadarStationName = str2Bytes('RadarStationName=%s,'%station)
    headinfoLen=20+len(DataName)+len(Year)+len(Month)+len(Day)+len(Hour)+len(Minute)+len(XNumGrids)+len(YNumGrids)+len(RadarCount)+len(StartLon)+len(StartLat)+len(EndLon)+len(EndLat)+len(XReso)+len(YReso)+len(RadarStationName)
    HeadinfoLen = str2Bytes('HeadinfoLen=%i     '%headinfoLen)
    try:
        os.makedirs("E:/product/"+generateTime+"/"+station+"/"+"bin")
    except:
        print('文件夹已经存在，无需新建')
    os.chdir("E:/product/"+generateTime+"/"+station+"/"+"bin")
    #写数据
    file = open(title+'.bin','wb')
    
    file.write(HeadinfoLen) 
    file.write(DataName)
    file.write(Year)
    file.write(Month)
    file.write(Day)
    file.write(Hour)
    file.write(Minute)
                    
    file.write(XNumGrids)
    file.write(YNumGrids)
    file.write(RadarCount)
    file.write(StartLon)
    file.write(StartLat)
    file.write(EndLon)
    file.write(EndLat)                     
    file.write(XReso)
    file.write(YReso)            
    file.write(RadarStationName)
    d=np.arange(gridH*gridC).reshape(gridH,gridC) 
    #根据行和列写入bin文件
    if len(crsave) == 0:
        for i in range(gridH):
            for j in range(gridC):
                d[i][j] = 0
                file.write(d[i,j])
        file.close()
        print('bin文件已经生成')
    else:
        for rc,result in zip(crsave,result_output):
            r = rc[0]
            c = rc[1]
            d[r,c] = result
            file.write(d[r,c])
        file.close()
        print('bin文件已生成') 

#画图像格点
def create_grid(crsave,result_output,path,gridH,gridC,station):
    filetime,mid_path,generateTime,image = empty_grid(crsave,result_output,path,gridH,gridC,station)
    for cr,result in zip(crsave,result_output):
        r = cr[0]
        c = cr[1]
        X=np.array([147.5+18*r,147.5+18*r,165.5+18*r,165.5+18*r])
        Y=np.array([517.5+16*c,533.5+16*c,533.5+16*c,517.5+16*c])
        rr, cc=draw.polygon(X,Y)
        if int(result) == 0:
            draw.set_color(image,[rr,cc],[255,255,255],0.6)
        elif int(result) == 1:
            draw.set_color(image,[rr,cc],[0,160,248],0.6)
        elif int(result) == 2:
            draw.set_color(image,[rr,cc],[255,0,0],0.6)
        elif int(result) == 3:
            draw.set_color(image,[rr,cc],[150,0,180],0.6)
    try:
        os.makedirs("E:/product/"+generateTime+"/"+station+"/"+"picture")
    except:
        print('文件夹已经存在，无需新建')
    os.chdir("E:/product/"+generateTime+"/"+station+"/"+"picture")
    title='Nowcasting_FindStorm'+'_'+filetime+'_'+station
    io.imsave(title+'.png',image)
    return filetime,generateTime,title

#初始化空图
def empty_grid(crsave,result_output,path,gridH,gridC,station):
    if len(path)-1 == 1:
        filetime = path[1].split(" ")[2]
        mid_path = filetime[:8]
        generateTime = mid_path[:4]+"-"+mid_path[4:6]+"-"+mid_path[6:8]
        image=io.imread(map_path)
        for i in range(gridH):
            for j in range(gridC):
                X=np.array([147.5+18*i,147.5+18*i,165.5+18*i,165.5+18*i])
                Y=np.array([517.5+16*j,533.5+16*j,533.5+16*j,517.5+16*j])
                rr, cc=draw.polygon(X,Y)
                draw.set_color(image,[rr,cc],[255,255,255],0.6)
        title='Nowcasting_FindStorm'+'_'+filetime+'_'+station
        try:
            os.makedirs("E:/product/"+generateTime+"/"+station+"/"+"picture")
        except:    
            print('文件夹已经存在，无需新建')
        os.chdir("E:/product/"+generateTime+"/"+station+"/"+"picture")
        io.imsave(title+'.png',image)
        return filetime,mid_path,generateTime,title
    else:
        filetime = path[1].split("_")[2]
        mid_path = filetime[:8]
        generateTime = mid_path[:4]+"-"+mid_path[4:6]+"-"+mid_path[6:8]
        image=io.imread(map_path)
        for i in range(gridH):
            for j in range(gridC):
                X=np.array([147.5+18*i,147.5+18*i,165.5+18*i,165.5+18*i])
                Y=np.array([517.5+16*j,533.5+16*j,533.5+16*j,517.5+16*j])
                rr, cc=draw.polygon(X,Y)
                draw.set_color(image,[rr,cc],[255,255,255],0.6)
        return filetime,mid_path,generateTime,image

    
    