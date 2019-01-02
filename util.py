#!/usr/bin/env python
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from skimage.feature import hog


#定义一个直方图
def color_hist(img, nbins=32, bin_range = (0,256)):
	rhist = np.histogram(img[:,:,0], bins = nbins, range = bins_range)
	ghist = np.histogram(img[:,:,1], bins = nbins, range = bins_range)
	bhist = np.histogram(img[:,:,2], bins = nbins, range = bins_range)

	bin_edges = rhist[1]
	bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1])/2

	hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))

	return rhist, ghist, bhist, bin_centers, hist_features


#计算直方图的特征
def bin_spatial(img, color_space = 'RGB', size = (32, 32)):
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space =='YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: 
		feature_image = np.copy(img)
	features = cv2.resize(feature_image, size).ravel()
	return features


#对数据集进行探索
def data_look(car_list, notcar_list):
	data_dict = {}
	data_dict['n_cars'] = len(car_list)
	data_dict['notcar_list'] = len(notcar_list)
	example_img = mpimg.imread(car_list[0])
	data_dict['image_shape'] = example_img.shape
	data_dict['data_type'] = example_img.dtype 
	return data_dict


#提取HOG特征
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis == True:
		features,hog_image = hog(img, orientations=orient, pixels_per_cell=(pixels_per_cell,pix_per_cell),
								cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
								visualise=True, feature_vector=False)
		return features, hog_image
	else:
		features = hog(img,orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell), 
						cells_per_block=(cell_per_block,cell_per_block), transform_sqrt=False,
						visualise=False, feature_vector=feature_vec)
		return features


#获取图像颜色特征
def extract_features_color(imgs, cspace='RGB', spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256)):

	features = []
	for file in imgs:
		image = mpimg.imread(file)
		if cspace != 'RGB':
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
		else:
			feature_image = np.copy(image)
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		hist_features = color_hist(feature_image, nbins=hist_bin, bins_range=hist_range)
		features.append(np.concatenate((spatial_features, hist_features)))

	return features


#计算图像特征	
def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):

	features = []
	for file in imgs:
		image = mpimg.imread(file)
		if cspace != 'RGB':
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else:
			feature_image = np.copy(image)

		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
                				
			hog_features = np.ravel(hog_features)
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel],orient,
											pix_per_cell, cell_per_block,vis = False, feature_vec=True)
		features.append(hog_features)
	return features


def convert_color(img, conv = 'RGB2YYCrCb'):
	if conv == 'RGB2YYCrCb':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	if conv == 'RGB2LUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def plot3d(pixels, colors_rgb, axis_labels=list('RGB'), axis_limits=((0, 255), (0, 255), (0, 255))):
	fig = plt.figure(figsize=(8, 8))
	ax = Axes3D(fig)

	ax.set_xlim(*axis_limits[0])
	ax.set_ylim(*axis_limits[1])
	ax.set_zlim(*axis_limits[2])

	ax.tick_params(axis='both', witch='major', labelsize=14, pad=8)
	ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
	ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
	ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

	ax.scatter(pixels[:,:,0].ravel(),
			   pixels[:,:,1].ravel(),
			   pixels[:,:,2].ravel(),
			   c=colors_rgb.reshape((-1, 3)), edgecolors='none')

	return ax


#图像处理	
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
				 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]

	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]

	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))

	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))

	nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
	ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

	window_list = []

	for ys in range(ny_windows):
		for xs in range(nx_windows):

			startx = xs * nx_pix_per_step + x_start_stop[0]
			endx =  startx + xy_window[0]

			starty = ys * ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]

			window_list.append(((startx, starty), (endx, endy)))

	return window_list


#获取单个图窗的特征
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
						spatial_feat=True, hist_feat=True, hog_feat=True):
	img_features = []

	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	else: 
		feature_image = np.copy(img)

	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)

		img_features.append(spatial_features)

	return np.concatenate(img_features)


#搜索图窗
def search_windows(img, windows, clf, scaler, color_space='RGB',
				   spatial_size=(32, 32), hist_bins=32,
				   hist_range=(0, 256), orient=9,
				   pix_per_cell=8, cell_per_block=2,
				   hog_channel=0, spatial_feat=True,
				   hist_feat=True, hog_feat=True):
	on_windows = []

	for window in windows:
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		#获取特征
		features = single_img_features(test_img, color_space=color_space,
										spatial_size=spatial_size, hist_bins=hist_bins,
										cell_per_block=cell_per_block,
										hog_channel=hog_channel, spatial_feat=spatial_feat,
										hist_feat=hist_feat, hog_feat=hog_feat)
		#特征缩放
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#预测
		prediction = clf.prediction(test_features)

		if prediction == 1:
			on_windows.append(window)
	return on_windows


def add_heat(heatmap, bbox_list):
	#遍历bboxes列表形成热图
	for box in bbox_list:
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	return heatmap


def apply_threshold(heatmap, threshold):
	#热图过滤
	heatmap[heatmap <= threshold] = 0

	return heatmap


def draw_labeled_bboxes(img, labels):
	for car_number in range(1, labels[1]+1):
		nonzero = (labels[0] == car_number).nonzero()

		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		#在图像中画图
		cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

	return img


def draw_windows(img, windows):
	draw_img = np.copy(img)
	for window in windows:
		cv2.rectangle(draw_img, windows[0], window[1], (0, 0, 255), 6)
	return draw_img






