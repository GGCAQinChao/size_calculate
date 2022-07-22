# coding: utf-8
# 在线工具网站地址：https://share.streamlit.io/ggcaqinchao/size_calculate/main/size_calculation.py
import re
import cv2
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from haishoku.haishoku import Haishoku

def RGB_to_Hex(rgb):
    #print(rgb)
    color = '#'
    for i in rgb:
        color += str(hex(int(i)))[-2:].replace('x','0').upper()
    return color

def pick_palette():
    st.title('主配色提取工具')
    image_path = st.text_input('请输入网络图片位置：')
    st.write('例：https://pic1.zhimg.com/80/v2-17099cb618b725367ec43d7b3654a248_720w.jpg')
    if st.button('RUN'):
        if image_path == '':
            st.write('啥图都没有诶，大兄弟！')
        else:
            st.image(image_path)
            haishoku = Haishoku.loadHaishoku(image_path)
            dominant = RGB_to_Hex(haishoku.dominant)
            color = st.color_picker('Dominant color:', dominant)
            palette = []
            col1,col2,col3,col4 = st.columns(4)
            for num in range(0,8):
                pal = haishoku.palette[num]
                palette.append([RGB_to_Hex(pal[1]),pal[0]])
            with col1:
                color = st.color_picker(palette[0][0]+'('+str(palette[0][1])+'):', palette[0][0])
                color = st.color_picker(palette[4][0]+'('+str(palette[4][1])+'):', palette[4][0])
            with col2:
                color = st.color_picker(palette[1][0]+'('+str(palette[1][1])+'):', palette[1][0])
                color = st.color_picker(palette[5][0]+'('+str(palette[5][1])+'):', palette[5][0])
            with col3:
                color = st.color_picker(palette[2][0]+'('+str(palette[2][1])+'):', palette[2][0])
                color = st.color_picker(palette[6][0]+'('+str(palette[6][1])+'):', palette[6][0])
            with col4:
                color = st.color_picker(palette[3][0]+'('+str(palette[3][1])+'):', palette[3][0])
                color = st.color_picker(palette[7][0]+'('+str(palette[7][1])+'):', palette[7][0])



def main():
    pick_palette()

if __name__ == '__main__':
    main()
