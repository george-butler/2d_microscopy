import numpy as np
import cv2
import pandas as pd
from glob import glob
import csv
import sys
import os

def contour_counter(cnt,x,y,cell):
    counter = 0
    cnt_len = 0
    k_count = 0
    for k in range(len(cnt)):
        cnt_n = cnt[k]
        if len(cnt_n) < 1000:
            d = cv2.pointPolygonTest(cnt_n,(x,y),False)
            if d > 0:
                counter = counter + 1
                if len(cnt_n) > cnt_len:
                    cnt_len = len(cnt_n)
                    k_count = k
    return counter,k_count


os.chdir("~/Extract_contours/Masks_per_frame")
a = []
b = []
c = []
d = []
dataframe = pd.read_csv("tracks.csv")
dataframe.apply(pd.to_numeric)
dataframe.sort_values(["frame","particle"], ascending=[True,True])
pngs = sorted(glob("./*.png"))
for index, i in enumerate(pngs):
    print(i)
    im = cv2.imread(i)
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    df = dataframe[dataframe.frame == index]
    df = df.reset_index()
    for j in range(len(df)):
        x = df["wtd_x"][j]
        y = df["wtd_y"][j]
        cell = df["particle"][j]
        for k in range(256):
            lower = np.array([k,1,0])
            upper = np.array([k,255,255])
            mask = cv2.inRange(hsv,lower,upper)
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) != 0:
                count,k_id = contour_counter(contours,x,y,cell)
                if count == 1:
                    n_contour = contours[k_id]
                    for m in range(len(n_contour)):
                        XY_Coordinates = n_contour[m]
                        a.append(XY_Coordinates[0][0])
                        b.append(XY_Coordinates[0][1])
                        c.append(cell)
                        d.append(index)

a = np.array(a)
b = np.array(b)
c = np.array(c)
d = np.array(d)
e = np.array([1] * len(a)) #video_number
f = np.array([1] * len(a)) #run number
g = np.array([1] * len(a)) #video key



datal = list(zip(a,b,d,c,e,f,g))
df1 = pd.DataFrame(data = datal)
df1.columns = ["x","y","frame_id","cell_id","video_number","run_number","video_key"]
df1.to_csv("xy1_12hour_r1.csv", index = False)
