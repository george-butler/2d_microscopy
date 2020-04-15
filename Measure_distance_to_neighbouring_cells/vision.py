import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
from glob import glob


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


def worker (a,b):
	results = []
	for i in range(len(a.index)):
		ax = a['x'][i]
		ay = a['y'][i]
		bx = np.array(b['x'])
		by = np.array(b['y'])
		distance = dist(ax,ay,bx,by)
		order_values = np.argsort(distance)

		for k in order_values:
			output = vision(ax,ay,b['x'][k],b['y'][k],a)
			if output == False:
				results.append([ax,ay,b['x'][k],b['y'][k],b['cell_id'][k]])
				break
	return results

def order(df,i):
	frame_n = df[df.frame_id == i]
	results = []
	cells = frame_n['cell_id'].unique()
	for j in cells:
		a = frame_n[frame_n.cell_id == j]
		b = frame_n[frame_n.cell_id != j]
		a = a.reset_index(drop = True)
		b = b.reset_index(drop = True)
		holder = worker(a,b)
		holder.extend([j,i])
		results.append(holder)
	return results
##########################################################################################################################
def main(a):
	dataframe = pd.read_csv(a)
	dataframe.apply(pd.to_numeric)

	frames = dataframe['frame_id'].unique()
	results_overall = Parallel(n_jobs = 28)(delayed(order)(dataframe,i) for i in frames)

	frame_id = []
	cell_id = []
	cell_x = []
	cell_y = []
	neighbor_x = []
	neighbor_y = []
	neighbour_id = []
	for i in range(len(results_overall)):
		for j in range(len(results_overall[i])):
			length = len(results_overall[i][j])
			frame_id.extend(([results_overall[i][j][(length-1)]] * (length-3)))
			cell_id.extend(([results_overall[i][j][(length-2)]] * (length-3)))
			for k in range(length-3):
				cell_x.append(results_overall[i][j][k][0])
				cell_y.append(results_overall[i][j][k][1])
				neighbor_x.append(results_overall[i][j][k][2])
				neighbor_y.append(results_overall[i][j][k][3])
				neighbour_id.append(results_overall[i][j][k][4])

	datal = list(zip(frame_id,cell_id,cell_x,cell_y,neighbor_x,neighbor_y,neighbour_id))
	df = pd.DataFrame(data = datal)
	df.columns=['frame_id','cell_id','cell_x','cell_y','neighbor_x','neighbor_y','neighbor_id']

	holder = []
	holder1 = []
	for index, row in df.iterrows():
		d = dist(row['cell_x'],row['cell_y'],row['neighbor_x'],row['neighbor_y'])
		holder.append(d)
		holder1.append(1/d)

	df['distance_to_neighbor']= holder
	df['inverse_dist']=holder1
	df["video_number"]=np.array([dataframe['video_number'][0]] * len(holder))
	df["run_number"]=np.array([dataframe["run_number"][0]] * len(holder))
	df["video_key"]=np.array([dataframe["video_key"][0]] * len(holder))

	df.to_csv(a[:-4]+str("_neighbour.csv"), index = False)


f = glob("*.csv")
for i in f:
	print(i)
	main(i)
