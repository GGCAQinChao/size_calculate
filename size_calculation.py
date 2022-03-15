# coding: utf-8
import re
import cv2
import streamlit as st
import time
import pandas as pd
import numpy as np
import os
from PIL import Image

def upload_and_show():
    st.write('图片读取后，由于不同的解码方式，可能会存在一定的色差，但是并不影响后续的计算操作，请放心使用。')
    upload_file = st.file_uploader('Choose a file')
    if upload_file is not None:
        bytes_data = upload_file.getvalue()
        ori_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.image(ori_image)
    else:
        ori_image = 'NA'
    return ori_image

def covert_to_gray(ori_image):
    st.write('将原始图像转为灰度图以用于后续的二值化操作，与此同时也提供R、G、B三通道的灰度图，在这里可以选择统计样本与背景差异最大的那个灰度图用于后续分析。在样本面积计算中，主要是通过圈定黑背景下的白色图块来计算的，大家可以根据实际情况选择是否进行颜色反转。')
    gray_img = cv2.cvtColor(ori_image,cv2.COLOR_BGR2GRAY)
    blue_img,green_img,red_img=cv2.split(ori_image)
    #cv2.imwrite('test_out0.blue.jpg',blue_img)
    #cv2.imwrite('test_out0.green.jpg',green_img)
    #cv2.imwrite('test_out0.red.jpg',red_img)
    col1,col2 = st.columns(2)
    with col1:
        st.subheader('RGB2GRAY')
        st.image(gray_img)
        st.subheader('R通道')
        st.image(red_img)
    with col2:
        st.subheader('G通道')
        st.image(green_img)
        st.subheader('B通道')
        st.image(blue_img)
    return gray_img,red_img,green_img,blue_img

def binarization(gray_list):
    img_gray = {'RGB2GRAY':0,'R通道':1,'G通道':2,'B通道':3}
    col1,col2 = st.columns(2)
    with col1:
        img_select = st.selectbox('灰度图选择：',['RGB2GRAY','R通道','G通道','B通道'])
    with col2:
        transfer_flag = st.checkbox('是否颜色反转')
    st.subheader('二值化和去噪')
    st.write('二值化阈值一般选择自动，但是也可以自定义，去噪阈值即是指你想要过滤的噪点的大小。')
    col1,col2 = st.columns(2)
    with col1:
        threshold_select = st.select_slider('二值化阈值：',options=['auto']+list(range(1,255)))
    with col2:
        size_select = st.slider('去噪阈值：',1,10,3)
    #灰度图选择
    img_use = gray_list[img_gray[img_select]]
    #二值化
    if transfer_flag:
        img_use = 255 - img_use
    if threshold_select == 'auto':
        thresh, binary_img = cv2.threshold(img_use,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        thresh, binary_img = cv2.threshold(img_use,threshold_select,255,cv2.THRESH_BINARY)
    #去噪
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(size_select,size_select))
    dst_img=cv2.morphologyEx(binary_img,cv2.MORPH_OPEN,element)
    col1,col2 = st.columns(2)
    with col1:
        st.subheader('二值化结果')
        st.image(binary_img)
    with col2:
        st.subheader('去噪后')
        st.image(dst_img)
    return dst_img

def size_calculate(dst_img,ori_image):
    st.write('基于二值化结果进行目的样本的圈定和面积计算，在这里可以调整对样本的检测阈值，从而过滤掉在前面去噪步骤没有去除的非目的噪声。圈画边框仅用作标注样本，并不意味着面积计算结果是源于直接计算边框的矩形面积。')
    col1,col2,col3 = st.columns(3)
    with col1:
        size_select = st.slider('面积小于多少不予计算：',1,1000,50)
    with col2:
        show_select = st.selectbox('展示边框还是轮廓:',['边框','轮廓'])
    with col3:
        color = st.color_picker('边框和字体颜色：', '#000000')
        value = color.lstrip('#')
        lv = len(value)
        rgb_color = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    col1,col2,col3 = st.columns(3)
    with col1:
        line_select = st.slider('圈画边框的粗细：',1,10,1)
    with col2:
        font_select = st.slider('编号字体的大小：',min_value=0.0, max_value=10.0, value=1.0, step=0.2)
    with col3:
        fonl_select = st.slider('编号字体的粗细：',1,10,1)
    contours, hierarchy = cv2.findContours(dst_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(dst_img, contours, -1,(120,0,0),2)
    #cv2.drawContours(ori_image, contours, -1,(120,0,0),2)
    size_dir = {'id':[],'size':[]}
    count=0
    ares_avrg=0
    for cont in contours:
        ares = cv2.contourArea(cont)
        if ares < size_select:
            continue
        count += 1
        rect = cv2.boundingRect(cont)
        size_dir['id'].append(count)
        size_dir['size'].append(float(ares))
        #矩形绘制
        if show_select == '边框':
            cv2.rectangle(ori_image,rect,rgb_color,line_select)
        else:
            cv2.drawContours(ori_image,cont, -1,rgb_color,line_select)
        y=10 if rect[1] < 10 else rect[1]
        #给愈伤编号
        cv2.putText(ori_image,str(count),(rect[0],y),cv2.FONT_HERSHEY_COMPLEX,font_select,rgb_color,fonl_select)
    col1,col2 = st.columns([2,1])
    with col1:
        st.subheader('样本圈画')
        st.image(ori_image)
    with col2:
        st.subheader('面积计算')
        st.dataframe(pd.DataFrame(size_dir))
    st.subheader('样本轮廓')
    st.image(dst_img)

def size_calculation():
    st.title('Pick and calculate size of callus or seeds')
    #load_ori_image
    st.header('原始图像读取')
    ori_image = upload_and_show()
    if ori_image == 'NA':
        st.write('No image.')
    else:
        #covert
        st.header('图像转换和二值化')
        gray_list = covert_to_gray(ori_image)
        #two
        dst_img = binarization(gray_list)
        #size calculation
        st.header('轮廓检测和面积计算')
        size_calculate(dst_img,ori_image)

def main():
    tool_select = st.sidebar.selectbox('Tools to select:',['size_calculate'])
    if tool_select == 'size_calculate':
        size_calculation()

if __name__ == '__main__':
    main()
