import numpy as np
import pandas as pd
from glob import glob
import cv2
import math
def translation(cnt,a):
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	translation_centrex = a/2
	translation_centrey = a/2
	movex = translation_centrex - cx
	movey = translation_centrey - cy
	cnt[:,:,0] = cnt[:,:,0] + movex
	cnt[:,:,1] = cnt[:,:,1] + movey
	return cnt

def contourBuilder(x,y):
    size = len(x)
    results = np.zeros(shape=(size,1,2)).astype(np.int32)
    idx = range(0,size)
    for i,j,k in zip(idx,x,y):
	    results[i]=[j,k]
    return results
def dist(x1,y1,x2,y2):
    distance = ((x1-x2)**2 + (y1 - y2)**2)**0.5
    return distance

def scaling(cnt,R,avg_radius):
    M = cv2.moments(cnt)
    xcentroid = int(M['m10']/M['m00'])
    ycentroid = int(M['m01']/M['m00'])
    distsum = avg_radius
    scale = R / distsum
    cnt[:,:,0] = scale * (cnt[:,:,0] - xcentroid) + xcentroid
    cnt[:,:,1] = scale * (cnt[:,:,1] - ycentroid) + ycentroid
    return cnt

def max_dist_cells(data, R,avg_radius):
    max_radius = 0
    for i in data['frame_id'].unique():
        df = data[data.frame_id == i]
        for j in df['cell_id'].unique():
            testx = df[df.cell_id == j]['x']
            testy = df[df.cell_id == j]['y']

            contour = contourBuilder(testx,testy)
            scale_contour = scaling(contour, R,avg_radius)


            M = cv2.moments(scale_contour)
            if M["m00"] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                coord = np.array(list(zip(scale_contour[:,:,0],scale_contour[:,:,1])))
                distance = max(dist(cx,cy,coord[:,0],coord[:,1]))
                if distance > max_radius:
                    max_radius = distance
    return max_radius
def average_radius(dataframe):
    mean_dist_per_cell = []
    dead_cells = []
    for i in dataframe['frame_id'].unique():
        df = dataframe[dataframe.frame_id == i]
        for j in df['cell_id'].unique():
            testx = df[df.cell_id == j]['x']
            testy = df[df.cell_id == j]['y']
            cnt = contourBuilder(testx,testy)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                xcentroid = int(M['m10']/M['m00'])
                ycentroid = int(M['m01']/M['m00'])
                size = len(cnt)
                distsum = []
                for k in range(size):
                    distance = dist(cnt[k,0,0],cnt[k,0,1],xcentroid,ycentroid)
                    distsum.append(distance)
                mdist = np.mean(distsum)
                mean_dist_per_cell.append(mdist)
            if M['m00'] == 0:
                dead_cells.append(j)
    return np.mean(mean_dist_per_cell),dead_cells

def roundup(x):
    return int(math.ceil(x/10.0)) * 10

f = glob("*.csv")

final_max_radius_dist = 0
final_image_size = 0
final_avg_radius = 0
avg_radius_holder = []

for i in f:
    dataframe = pd.read_csv(i)
    dataframe.apply(pd.to_numeric)
    print(i)

    R = 50
    print("R",R)
    D = 20 

    avg_radius,dead_cells = average_radius(dataframe)

    avg_radius_holder.append(avg_radius)

    dataframe = dataframe[~dataframe.cell_id.isin(dead_cells)]

    max_radius_dist = max_dist_cells(dataframe,R,avg_radius)[0]

    img_size = roundup(max_radius_dist)*2


    if max_radius_dist > final_max_radius_dist:
        final_max_radius_dist = max_radius_dist
    if img_size > final_image_size:
        final_image_size = img_size

print("avg_radius",sum(avg_radius_holder)/len(avg_radius_holder))
print("final_image_size",final_image_size)