from __future__ import print_function
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from glob import glob
import pandas as pd
import os
from skimage.io import imread
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def create_video_stream(in_file, time_split):
    """
    Create an output at a given interval
    Parameters
    -------
    in_file : string or integer (webcam)
    time_split :  Time between frames
    """
    cap = cv2.VideoCapture(in_file)
    start_time = time.time()
    get_time = time_split
    while(cap.isOpened()):
        ret, frame = cap.read()
        if isinstance(in_file,str):
            # get time code from video file
            cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        else:
            cur_time = time.time()-start_time
        
        if cur_time>=get_time:
            get_time += time_split
            yield frame, cur_time
    cap.release()
    cv2.destroyAllWindows()
# ts function
from bolt.spark.array import BoltArraySpark


def np_rdd_to_bolt(in_rdd):
    f_key, f_val = in_rdd.first()
    bolt_shape = tuple([in_rdd.count()] + list(f_val.shape))
    out_bolt = BoltArraySpark(rdd = in_rdd, shape = bolt_shape, split = 1, dtype = f_val.dtype)
    out_bolt._mode = '4Quant IQAE Engine'
    out_bolt.flatten = lambda : out_bolt.reshape((out_bolt.shape[0], np.prod(out_bolt.shape[1:])))
    return out_bolt

def show_row_img(cur_row, i = 0):
	c_path = cur_row['path'].to_dict().values()[i]
	fig, ax1 = plt.subplots(1,1, figsize = (15, 15))
	ax1.imshow(imread(c_path), cmap = plt.cm.GnBu, interpolation = 'lanczos')
	ax1.axis('off')

def read_structured_data(data_dir, file_path):
    file_df = pd.read_csv(os.path.join(data_dir,file_path), sep = '\t')
    file_df['path'] = file_df['Image No.'].map(lambda i: os.path.join(data_dir,'Archive','141110A3.%04d' % i))
    return file_df

def read_stack(sc, file_df):
	all_images = sc.parallelize([(ij['Time(hrs)'], ij['path']) for ij in file_df.sort_values(by='Time(hrs)').T.to_dict().values()]).filter(lambda (_, path): os.path.exists(path)).sortByKey()
	full_stack = all_images.mapValues(imread)
	prebolt_stack = full_stack.zipWithIndex().map(lambda ((_, img), i): ((i,),img))
	time_axis = all_images.map(lambda (time_val, _): time_val).collect()
	return  time_axis, np_rdd_to_bolt(prebolt_stack)

make_base_img = lambda bt_stack, i=0: plt.cm.GnBu((bt_stack[i,:,:].toarray()/7000.0).clip(0,1))

def show_time_series(base_img, block_data, time_axis, x_pos, y_pos):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))
    cur_img = base_img.copy()
    cur_img[x_pos, : , 0] = 1.0
    cur_img[x_pos, : , 1:2] = 0
    cur_img[:, y_pos , 0] = 1.0
    cur_img[:, y_pos , 1:2] = 0.0
    ax1.imshow(cur_img)
    ax1.set_title('Image Preview')
    ax1.axis('off')
    ax2.plot(time_axis,block_data[:,x_pos, y_pos].toarray())
    ax2.set_title('Time Series Plot')
    ax2.set_xlabel('Time (Hours)')
    ax2.set_ylabel('Intensity (au)')
    
    
def meshgridnd_like(in_img, rng_func = range):
    """
    Makes a n-d meshgrid in the shape of the input image
    """
    new_shape = list(in_img.shape)
    fixed_shape = [new_shape[1], new_shape[0]]+new_shape[2:] if len(new_shape)>=2 else new_shape 
    all_range = [rng_func(i_len) for i_len in fixed_shape]
    return np.meshgrid(*all_range)



from sklearn.cross_decomposition import PLSRegression as PartialLSQ
from sklearn.cluster import KMeans as kMeans

def extract_conc_vec(in_stack, group_cnt = 6):
    xx, yy = meshgridnd_like(in_stack[0])
    conc_vec = np.stack([xx.flatten(), yy.flatten()],1).astype(np.float32)
    kms = kMeans(group_cnt)
    kms.fit(conc_vec)
    return kms.predict(conc_vec)

def fit_pls(in_pls, in_x, in_y):
    in_pls.fit(in_x.flatten().T.toarray(), in_y)
    
def transform_pls(in_pls, in_x):
    return in_pls.transform(in_x.flatten().T.toarray())
def predict_pls(in_pls, in_x):
    return in_pls.predict(in_x.flatten().T.toarray())