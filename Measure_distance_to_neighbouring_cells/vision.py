import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
from glob import glob
import joblib 

def unpacking_data(t):
    dataframe = np.empty((0,t[0].shape[1]))
    for i in t:
        dataframe = np.vstack([dataframe,i])
    df = pd.DataFrame(dataframe, columns = ['frame_id','cell_id','cell_x','cell_y','neighbor_x','neighbor_y','neighbor_id'])
    return df


def dist(x1,y1,x2,y2):
	distance = ((x1-x2)**2 + (y1 - y2)**2)**0.5
	return distance


def orientation(p,q,r):

	value = ((q[1] - p[1])*(r[0] - q[0])) - ((q[0] - p[0])*(r[1] - q[1]))

	if value == 0:
		return 0

	if value > 0:
		return 1
	if value < 0:
		return 2

def onSegment(p ,q ,r):

	if (q[0] <= max(p[0],r[0])) and (q[0] >= min(p[0],r[0])) and (q[1] <= max(p[1],r[1])) and (q[1] >= min(p[1],r[1])):
		return True
	else:
		return False

def doIntersection(p1,q1,p2,q2):

	o1  = orientation(p1,q1,p2)
	o2  = orientation(p1,q1,q2)
	o3  = orientation(p2,q2,p1)
	o4  = orientation(p2,q2,q1)

	if (o1 != o2) and (o3 != o4):
		return True

	if (o1 == 0 and onSegment(p1, p2, q1)):
		return True
	if (o2 == 0 and onSegment(p1, q2, q1)):
		return True
	if (o3 == 0 and onSegment(p2, p1, q2)):
		return True
	if (o4 == 0 and onSegment(p2, q1, q2)):
		return True

	return False

def vision (ax,ay,bx,by,a):
	holder = False
	xy = list(zip(a['x'],a['y']))
	for i in range(len(xy)):
		c = xy[i]
		if i == (len(xy) - 1):
			d = xy[0]
		else:
			d = xy[(i+1)]
		if ((ax,ay) != c and (ax,ay) != d):
			holder = doIntersection((ax,ay),(bx,by),c,d)
			if holder == True:
				break
	return holder


def worker (a,b,frame,cell):
	results = np.empty((0,7))
	for i in range(len(a.index)):
		ax,ay = a['x'][i],a['y'][i]
		bx,by = np.array(b['x']),np.array(b['y'])
		distance = dist(ax,ay,bx,by)
		order_values = np.argsort(distance)
		for k in order_values:
			output = vision(ax,ay,b['x'][k],b['y'][k],a)
			if output == False:
				d = np.column_stack([frame,cell,ax,ay,b['x'][k],b['y'][k],b['cell_id'][k]])
				results = np.vstack([results,d])
				break
	return results

def order(df,i):
	frame_n = df[df.frame_id == i]
	results = np.empty((0,7))
	cells = frame_n['cell_id'].unique()
	for j in cells:
		a = frame_n[frame_n.cell_id == j]
		b = frame_n[frame_n.cell_id != j]
		a = a.reset_index(drop = True)
		b = b.reset_index(drop = True)
		holder = worker(a,b,i,j)
		results = np.vstack([results,holder])
	return results
##########################################################################################################################
def main(a):
	dataframe = pd.read_csv(a)
	dataframe.apply(pd.to_numeric)

	frames = dataframe['frame_id'].unique()
	num_cores = joblib.cpu_count()
	results_overall = Parallel(n_jobs = num_cores - 2, backend="threading",verbose =5 )(delayed(order)(dataframe,i) for i in frames)

	df = unpacking_data(results_overall)

	df.to_csv(a[:-4]+str("_neighbour.csv"), index = False)

	holder = []
	holder1 = []
	for index, row in df.iterrows():
		d = dist(row['cell_x'],row['cell_y'],row['neighbor_x'],row['neighbor_y'])
		holder.append(d)
		holder1.append(1/d)

	df['distance_to_neighbor']= holder
	df['inverse_dist']=holder1
	df.to_csv(a[:-4]+str("_neighbour.csv"), index = False)


f = glob("*.csv")
for i in f:
	print(i)
