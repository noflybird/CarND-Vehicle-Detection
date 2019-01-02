#!/usr/bin/env python
"""
使用训练好的模型
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np 
import pickle 
import cv2
import util
import glob
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

dist_pickle = pickle.load(open('train_dist.p', 'rb'))
svc = dist_pickle['clf']
X_scaler = dist_pickle['scaler']
orient = dist_pickle['orient']
pix_per_cell = dist_pickle['pix_per_cell']
cell_per_block = dist_pickle['cell_per_block']
spatial_size = dist_pickle['spatial_size']
hist_bins = dist_pickle['hist_bins']



class Vehicle_Detect():
    def __init__(self):
        # 帧历史直方图坐标
        self.prev_rects = [] 
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            # 丢弃部分旧直方图
            self.prev_rects = self.prev_rects[len(self.prev_rects)-15:]


#提取特征以及做预测
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler,
			 orient, pix_per_cell,cell_per_block, spatial_size, hist_bins, 
			 show_all_rectangles=False):
	windows = []
	img = img.astype(np.float32) / 255
	img_tosearch = img[ystart:ystop, :, :]

	if cspace != 'RGB':
		if cspace == 'HSV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
		if cspace == 'LUV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
		if cspace == 'HLS':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
		if cspace == 'YUV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
		if cspace == 'YCrCb':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
	else:
		ctrans_tosearch = np.copy(img)

	#重新缩放数据
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

	#为HOG方向
	if hog_channel == 'ALL':
		ch1 = ctrans_tosearch[:, :, 0]
		ch2 = ctrans_tosearch[:, :, 1]
		ch3 = ctrans_tosearch[:, :, 2]
	else:
		ch1 = ctrans_tosearch[:, :, hog_channel]

	#定义直方图区间和步长
	nxblocks = (ch1.shape[1] // pix_per_cell) + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) + 1
	nfeat_per_block = orient * cell_per_block ** 2
	#原始采样率
	window = 64
	nblocks_per_window = (window // pix_per_cell) - 1
	cells_per_step = 2
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	#计算全部图像的HOG特征
	hog1 = util.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	if hog_channel == 'ALL':
		hog2 = util.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog3 = util.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb * cells_per_step
			xpos = xb * cells_per_step

			#从当前块提取HOG
			hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
			if hog_channel == 'ALL':
				hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
				hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
			else:
				hog_features = hog_feat1
			xleft = xpos * pix_per_cell
			ytop = ypos * pix_per_cell

			test_prediction = svc.predict(hog_features)

			if test_prediction == 1 or show_all_rectangles:
				xbox_left = np.int(xleft * scale)
				ytop_draw = np.int(ytop * scale)
				win_draw = np.int(window * scale)
				windows.append((
					(xbox_left, ytop_draw + ystart), 
					(xbox_left + win_draw, ytop_draw + win_draw + ystart)
					))
	return windows


def search_car(img):
	draw_img = np.copy(img)

	windows = []

	colorspace = 'YUV'
	orient = 11
	pix_per_cell = 16
	cell_per_block = 2
	hog_channel = 'ALL'

	ystart = 400
	ystop = 464
	scale = 1.0
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	ystart = 416
	ystop = 480
	scale = 1.0
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	ystart = 400
	ystop = 496
	scale = 1.5
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	ystart = 432
	ystop = 528
	scale = 1.5
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	ystart = 400
	ystop = 528
	scale = 2.0
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	ystart = 432
	ystop = 560
	scale = 2.0
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	ystart = 400
	ystop = 596
	scale = 3.5
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	ystart = 464
	ystop = 660
	scale = 3.5
	windows.append(find_cars(
					img, ystart, ystop, scale, colorspace, hog_channel, svc,
					None, orient, pix_per_cell, cell_per_block, None, None
					))

	#window_list = util.slide_window(img)
	rect = [j for i in windows for j in i]
	heat_map = np.zeros(img.shape[:2])
	if len(rect) > 0:
		det.add_rects(rect)
	for rect_set in det.prev_rects:
		heatmap = util.add_heat(heatmap, rect_set)
	heatmap = util.apply_threshold(heatmap, 1 + len(det.prev_rects) // 2)
	labels = label(heatmap)
	draw_img = util.draw_labeled_bboxes(draw_img, labels)

	return draw_img



# test_imgs = []
# out_imgs = []
# img_paths = glob.glob('test_image/*.jpg')
# for path in img_paths:
# 	img = mpimg.imread(path)
# 	out_img = search_car(img)
# 	test_imgs.append(img)
# 	out_imgs.append(out_imgs)

# plt.figure(figsize=(20, 68))
# for i in range(len(test_imgs)):
# 	plt.subplot(2*len(test_imgs), 2, 2*i+1)
# 	plt.imshow(test_imgs[i])

# 	plt.subplot(2*len(test_imgs), 2, 2*i+1)
# 	plt.imshow(out_imgs[i])



det = Vehicle_Detect()

project_outpath = 'video_out/project_video_out4.mp4'
project_video_clip = VideoFileClip('project_video.mp4')
project_video_out_clip = project_video_clip.fl_image(search_car)
project_video_out_clip.write_videofile(project_outpath, audio=False)
















