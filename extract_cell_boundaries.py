# coding: utf-8
# 在线工具网站地址：https://share.streamlit.io/ggcaqinchao/some_little_tools/main/extract_cell_boundaries.py
import re
import cv2
import streamlit as st
import time
import pandas as pd
import numpy as np
import os
from PIL import Image
from skimage import morphology

def upload_and_show():
    st.write('图片读取后，由于不同的解码方式，可能会存在一定的色差，但是并不影响后续的计算操作，请放心使用。')
    upload_file = st.file_uploader('Choose a file')
    if upload_file is not None:
        bytes_data = upload_file.getvalue()
        ori_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.image(ori_image,output_format='PNG')
    else:
        ori_image = 'NA'
    return ori_image

def pick_skeleton():
    Image.MAX_IMAGE_PIXELS = None
    st.header('原始图像读取')
    ori_image = upload_and_show()
    if ori_image == 'NA':
        st.write('No image.')
    else:
        gray = cv2.cvtColor(ori_image,cv2.COLOR_BGR2GRAY)
        st.header('参数设置')
        col1,col2,col3 = st.columns(3)
        with col1:
            threshold_min = st.select_slider('二值化阈值最小值：',list(range(1,255)),50)
        with col2:
            threshold_max = st.select_slider('二值化阈值最大值：',list(range(1,255)),200)
        with col3:
            threshold_level = st.select_slider('二值化阈值梯度：',list(range(1,255)),10)
        if threshold_min >= threshold_max or threshold_level >= threshold_max-threshold_min:
            st.write('二值化阈值设定错误.')
        elif st.button('Run'):
            skeleton0_list = []
            st.header('骨架提取结果')
            element = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
            col1,col2 = st.columns(2)
            for num in range(threshold_min,threshold_max+1,threshold_level):
                ret,binary = cv2.threshold(gray,num,255,cv2.THRESH_BINARY)
                col1,col2 = st.columns(2)
                with col1:
                    st.subheader('binary-'+str(num))
                    st.image(binary,output_format='PNG')
                binary[binary == 255] = 1
                skeleton0 = morphology.skeletonize(binary)
                skeleton_out = skeleton0.astype(np.uint8)*255
                skeleton0_list.append(skeleton0.astype(np.uint8))
                with col2:
                    st.subheader('skeleton-'+str(num))
                    st.image(skeleton_out,output_format='PNG')
            skeleton_sum = sum(skeleton0_list)*int(255/((threshold_max-threshold_min)/threshold_level))
            skeleton_sum = cv2.morphologyEx(skeleton_sum,cv2.MORPH_OPEN,element)
            st.subheader('骨架图合并')
            st.image(skeleton_sum,output_format='PNG')

def main():
    st.sidebar.header('模块选择')
    tool_select = st.sidebar.selectbox('Tools to select:',['细胞骨架提取'])
    if tool_select == '细胞骨架提取':
        pick_skeleton()

if __name__ == '__main__':
    main()
    
