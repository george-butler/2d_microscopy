import numpy as np
import pandas as pd
import cv2
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from math import atan2
from numpy import cos, sin, conjugate, sqrt, pi
from joblib import Parallel, delayed
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
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            coord = np.array(list(zip(scale_contour[:,:,0],scale_contour[:,:,1])))
            distance = max(dist(cx,cy,coord[:,0],coord[:,1]))
            if distance > max_radius:
                max_radius = distance
    return max_radius
def roundup(x):
    return int(math.ceil(x/10.0)) * 10
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
def zernike_reconstruct(img, radius, cof):

    idx = np.ones(img.shape)

    cofy,cofx = cof
    cofy = float(cofy)
    cofx = float(cofx)
    radius = float(radius)

    Y,X = np.where(idx > 0)

    P = img[Y,X].ravel()


    Yn = ( (Y -cofy)/radius).ravel()
    Xn = ( (X -cofx)/radius).ravel()

    k = (np.sqrt(Xn**2 + Yn**2) <= 1.)

    frac_center = np.array(P[k], np.float)

    Yn = Yn[k]
    Xn = Xn[k]
    frac_center = frac_center.ravel()
    npix = float(frac_center.size)

    return Xn,Yn,npix,frac_center


def main_pre(f_name):
    dataframe = pd.read_csv(f_name)
    dataframe.apply(pd.to_numeric)

    R = 50
    print("R",R)
    D = 20

    avg_radius,dead_cells = average_radius(dataframe)
    avg_radius = 21.30090250935587

    dataframe = dataframe[~dataframe.cell_id.isin(dead_cells)]


    print("Max Moment Order",D)
    starter = D
    print("Moment order",starter)

    #max_radius_dist = max_dist_cells(dataframe,R,avg_radius)[0]
    max_radius_dist = 452.20570540407823

    #img_size = roundup(max_radius_dist)*2
    img_size = 920

    print("IMG_SIZE", img_size)

    newx = []
    newy = []

    cell_counter = []
    for i in dataframe['frame_id'].unique():
        df = dataframe[dataframe.frame_id == i]
        for j in df['cell_id'].unique():
            cell_counter.append(j)
            testx = df[df.cell_id == j]['x']
            testy = df[df.cell_id == j]['y']

            contour = contourBuilder(testx,testy)
            trans_contour = translation(contour,img_size)
            scale_contour = scaling(trans_contour,R,avg_radius)

            newx.extend([i[0] for i in scale_contour[:,:,0]])
            newy.extend([i[0] for i in scale_contour[:,:,1]])
    dataframe['newx'] = newx
    dataframe['newy'] = newy


    with open("rad_poly.txt", "w") as out_file:
        framez = dataframe['frame_id'].unique()
        for i in framez:
            print(i)
            df = dataframe[dataframe.frame_id == i]

            if i == 0:
                no_1 = df['cell_id'].unique()[0]

            for j in df['cell_id'].unique():
                testx = df[df.cell_id == j]['newx']
                testy = df[df.cell_id == j]['newy']

                contour = contourBuilder(testx,testy)

                mask = np.zeros((img_size,img_size),np.uint8)
                img = cv2.drawContours(mask, [contour], -1, (255),-1)
                rows,cols = img.shape
                radius = cols//2 if rows > cols else rows//2

                xx,yy,nn,ff = zernike_reconstruct(img,radius,(rows/2.,cols/2.))

                if ((i==0) and (j ==no_1)):
                    z = np.vstack((xx,yy)).ravel('F')
                    h1 = np.round(z,3)
                    d_np = np.array(str(D))
                    h11 = np.append(d_np,h1)
                    np.savetxt(out_file, h11[np.newaxis], fmt='%s', delimiter='\t')

                v = np.array([i,j,nn])
                x = np.nonzero(ff)[0]
                h = np.append(v,x)
                np.savetxt(out_file, h[np.newaxis],fmt='%d',delimiter='\t')
