import numpy as np
import pandas as pd

def main_post(s_name,orig_data):
    D = 20
    print("Max Moment Order", D)

    d = np.genfromtxt("moments.txt", delimiter = "\t")[:,:-1]

    frame = []
    cell = []
    moment = []
    for i in range(len(d)):
        f = d[i][0]
        c = d[i][1]
        m = d[i][2:]

        ff = [f] * len(m)
        cc = [c] * len(m)

        frame.append(ff)
        cell.append(cc)
        moment.append(m)


    frame_flat = [item for sublist in frame for item in sublist]
    cell_flat = [item for sublist in cell for item in sublist]
    moment_flat = np.array([item for sublist in moment for item in sublist]) * 255


    data_l = list(zip(frame_flat,cell_flat,moment_flat))
    df = pd.DataFrame(data = data_l)
    df.columns = ["frame_id","cell_id","moment_value"]


    a = []
    for i in range(D+1):
        a.append(i)
    even = []
    odd = []
    for i in a:
        if i % 2 == 0:
            even.append(i)
        if i % 2 != 0:
            odd.append(i)
    even_zm = []
    even_zn = []
    for i in even:
        for j in even:
            if j <= i:
                even_zm.append(i)
                even_zn.append(j)
    odd_zm = []
    odd_zn = []
    for i in odd:
        for j in odd:
            if j <= i:
                odd_zm.append(i)
                odd_zn.append(j)
    evenl = list(zip(even_zm,even_zn))
    oddl = list(zip(odd_zm,odd_zn))
    totall = evenl + oddl

    df_index = pd.DataFrame(data = totall)
    df_index.columns = ["moment","az_angle"]
    df_index_s = df_index.sort_values(["moment","az_angle"], ascending = [True,True])
    df_index_s = df_index_s[2:]
    df_2 = pd.concat([df_index_s] * len(d))
    df_2 = df_2.reset_index(drop = True)
    final = pd.concat([df, df_2],axis=1)
    original_dataframe = pd.read_csv(orig_data)
    final["video_number"]=np.array([original_dataframe['video_number'][0]] * len(final))
    final["run_number"]=np.array([original_dataframe["run_number"][0]] * len(final))
    final["video_key"]=np.array([original_dataframe["video_key"][0]] * len(final))
    final.to_csv(str(s_name)+"_moments_20.csv",index = False)
