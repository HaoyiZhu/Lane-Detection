#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------
# Copyright (c) HaoyiZhu, Shanghai Jiao Tong University. All rights reserved.
# Github: https://github.com/HaoyiZhu
# Email: zhuhaoyi@sjtu.edu.com
# Reference:
#   - https://blog.csdn.net/qq_43243022/article/details/89158972
#   - https://github.com/yang1688899/CarND-Advanced-Lane-Lines
# -----------------------------------------------------

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

"""----------------------------- Demo Options -----------------------------"""
parser = argparse.ArgumentParser(description='LaneDetection Demo')
parser.add_argument('--inputfile', dest='inputfile',
                    help='image-directory', default="resources/20200618 124512_outVideo.avi")
parser.add_argument('--outputfile', dest='outputfile',
                    help='output-filename', default="results/output.mp4")
parser.add_argument('--mode', type=str, default='hist',
                    help='to use opencv for simple detection or use hist approach, option: hist/simple')

args = parser.parse_args()

class LaneDetection():
    def __init__(self, args):
        self.mode = args.mode
        self.outputfile = args.outputfile
        self.inputfile = args.inputfile

        # SIMPLE (OPENCV)
        if self.mode == 'simple':
            self.cache = None
            self.first_frame = 1

            self.kernel_size = 5
            self.low_threshold = 50
            self.high_threshold = 150

            self.rho = 1
            self.theta = np.pi / 180
            self.threshold = 15
            self.min_line_len = 40
            self.max_line_gap = 20

        # HIST
        elif self.mode == 'hist':
            # _threshoding
            self.x_thresh = [5, 200]
            self.mag_thresh = (10, 130)
            self.dir_thresh = (0.5, 1.1)
            self.hls_thresh = (5, 50)

            # _get_M_Minv
            self.src = [(0, 500), (500, 350), (834, 350), (1280, 500)]
            self.dst = [(320, 720), (320, 0), (1000, 0), (1000, 720)]

            # _find_line
            self.nwindows = 9
            self.margin = 100
            self.minpix = 50

        else:
            raise NotImplementedError

    def _grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    def _region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def _draw_lines(self, img, lines, color=[255, 0, 0], thickness=6):
        if not (type(lines) == np.ndarray):
            return img
        left_lines, right_lines = [], []#用于存储左边和右边的直线
        for line in lines:#对直线进行分类
            for x1, y1, x2, y2 in line:
                k = (y2 - y1) / (x2 - x1)
                if k < 0:
                    left_lines.append(line)
                else:
                    right_lines.append(line)

        if (len(left_lines) <= 0 or len(right_lines) <= 0):
            return img

        l_exist = self._clean_lines(left_lines, 0.1)#弹出左侧不满足斜率要求的直线
        r_exist = self._clean_lines(right_lines, 0.1)#弹出右侧不满足斜率要求的直线

        if (l_exist <= 0 or r_exist <= 0):
            return img

        left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]#提取左侧直线族中的所有的第一个点
        left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]#提取左侧直线族中的所有的第二个点
        right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]#提取右侧直线族中的所有的第一个点
        right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]#提取右侧侧直线族中的所有的第二个点

        left_vtx = self._calc_lane_vertices(left_points, 325, img.shape[0])#拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
        right_vtx = self._calc_lane_vertices(right_points, 325, img.shape[0])#拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标
        lx, _ = left_vtx[1]
        rx, _ = right_vtx[1]

        if lx < img.shape[1] / 2:
            cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)#画出直线
        if rx > img.shape[1] / 2:
            cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)#画出直线

    def _clean_lines(self, lines, threshold):
        slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
        slope = [k for k in slope if abs(k) > 0.2]
        if len(slope) == 0:
            return -1
        while len(lines) > 0:
            mean = np.mean(slope)  # 计算斜率的平均值，因为后面会将直线和斜率值弹出
            diff = [abs(s - mean) for s in slope]  # 计算每条直线斜率与平均值的差值
            idx = np.argmax(diff)  # 计算差值的最大值的下标
            if diff[idx] > threshold:  # 将差值大于阈值的直线弹出
                slope.pop(idx)  # 弹出斜率
                lines.pop(idx)  # 弹出斜率
            else:
                break
        return 1

    def _calc_lane_vertices(self, point_list, ymin, ymax):
        x = [p[0] for p in point_list]  # 提取x
        y = [p[1] for p in point_list]  # 提取y
        fit = np.polyfit(y, x, 1)  # 用一次多项式x=a*y+b拟合这些点，fit是(a,b)
        fit_fn = np.poly1d(fit)  # 生成多项式对象a*y+b

        xmin = int(fit_fn(ymin))  # 计算这条直线在图像中最左侧的横坐标
        xmax = int(fit_fn(ymax))  # 计算这条直线在图像中最右侧的横坐标

        return [(xmin, ymin), (xmax, ymax)]

    def _hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self._draw_lines(line_img, lines)
        return line_img

    def _weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        """
        `img` is the output of the hough_lines(), An image with lines drawn on it.
        Should be a blank image (all black) with lines drawn on it.

        `initial_img` should be the image before any processing.

        The result image is computed as follows:

        initial_img * α + img * β + λ
        NOTE: initial_img and img must be the same shape!
        """
        return cv2.addWeighted(initial_img, α, img, β, λ)


    def _abs_sobel_thresh(self, img, orient='x', thresh_min=0, thresh_max=255):
        # 装换为灰度图片
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 使用cv2.Sobel()计算计算x方向或y方向的导数
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # 阈值过滤
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        return binary_output

    def _mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    def _hls_select(self, img, channel='s', thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if channel == 'h':
            channel = hls[:, :, 0]
        elif channel == 'l':
            channel = hls[:, :, 1]
        else:
            channel = hls[:, :, 2]
        binary_output = np.zeros_like(channel)
        binary_output[(channel > thresh[0]) & (channel <= thresh[1])] = 1
        return binary_output

    def _dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def _thresholding(self, img):
        x_thresh = self._abs_sobel_thresh(img, orient='x', thresh_min=self.x_thresh[0], thresh_max=self.x_thresh[1])
        mag_thresh = self._mag_thresh(img, sobel_kernel=3, mag_thresh=self.mag_thresh)
        dir_thresh = self._dir_threshold(img, sobel_kernel=3, thresh=self.dir_thresh)
        hls_thresh = self._hls_select(img, thresh=self.hls_thresh)

        # Thresholding combination
        threshholded = np.zeros_like(x_thresh)
        threshholded[
            ((x_thresh == 1) & (mag_thresh == 1)) | ((dir_thresh == 1) & (hls_thresh == 1))] = 1

        return threshholded

    def _get_M_Minv(self):
        src = np.float32([self.src])
        dst = np.float32([self.dst])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def _find_line(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = self.nwindows
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self.margin
        # Set minimum number of pixels found to recenter window
        minpix = self.minpix
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if (len(lefty) <= 2 or len(leftx) <= 2) and (len(righty) <= 2 or len(lefty) <= 2):
            return -1, -1, -1, -1
        # Fit a second order polynomial to each

        if len(righty) <= 2 or len(lefty) <= 2:
            left_fit = np.polyfit(lefty, leftx, 2)
            y_min = min(lefty)
            right_fit = np.polyfit([720, y_min / 2, y_min], [1280, 1280, 1280], 2)
        elif len(lefty) <= 2 or len(leftx) <= 2:
            right_fit = np.polyfit(righty, rightx, 2)
            y_min = min(righty)
            left_fit = np.polyfit([720, y_min / 2, y_min], [0, 0, 0], 2)
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, left_lane_inds, right_lane_inds

    def _calculate_curv_and_pos(self, binary_warped, left_fit, right_fit):
        # Define y-value where we want radius of curvature
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        y_eval = np.max(ploty)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        curvature = ((left_curverad + right_curverad) / 2)
        # print(curvature)
        lane_width = np.absolute(leftx[719] - rightx[719])
        lane_xm_per_pix = 3.7 / lane_width
        veh_pos = (((leftx[719] + rightx[719]) * lane_xm_per_pix) / 2.)
        cen_pos = ((binary_warped.shape[1] * lane_xm_per_pix) / 2.)
        distance_from_center = veh_pos - cen_pos
        return curvature, distance_from_center

    def _draw_area(self, undist, binary_warped, Minv, left_fit, right_fit):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        return result

    def process_an_image_simple(self, image):
        gray_image=self._grayscale(image)

        gauss_gray = self._gaussian_blur(gray_image, self.kernel_size)
        canny_edges = self.canny(gauss_gray, self.low_threshold, self.high_threshold)

        # trapezoid ROI
        imshape = image.shape
        lower_left = [0, imshape[0]]
        lower_right = [imshape[1], imshape[0]]
        top_left = [0, imshape[0] / 1.5]
        top_right = [imshape[1], imshape[0] / 1.5]
        vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        roi_image = self._region_of_interest(canny_edges, vertices)

        # Hough Transform
        line_image = self._hough_lines(roi_image, self.rho, self.theta, self.threshold, self.min_line_len, self.max_line_gap)

        result = self._weighted_img(line_image, image, α=0.8, β=1., λ=0.)

        return result

    def process_an_image_hist(self, image):
        thresholded = self._thresholding(image)

        # trapezoid ROI
        imshape = image.shape
        lower_left = [0, imshape[0]]
        lower_right = [imshape[1], imshape[0]]
        top_left = [0, imshape[0] / 1.75]
        top_right = [imshape[1], imshape[0] / 1.75]
        vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
        roi_image = self._region_of_interest(thresholded, vertices)

        M, Minv = self._get_M_Minv()
        thresholded_wraped = cv2.warpPerspective(roi_image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        left_fit, right_fit, left_lane_inds, right_lane_inds = self._find_line(thresholded_wraped)
        if (isinstance(left_fit, int) and left_fit == -1) and (isinstance(right_fit, int) and right_fit == -1):
            return image

        curvature, distance_from_center = self._calculate_curv_and_pos(thresholded_wraped, left_fit, right_fit)

        result = self._draw_area(image, thresholded_wraped, Minv, left_fit, right_fit)

        return result

    def output(self):
        clip1 = VideoFileClip(self.inputfile)  # 读入input video
        print(clip1.fps)  # frames per second 25, 默认传给write

        if self.mode == 'simple':
            process_an_image = self.process_an_image_simple
        elif self.mode == 'hist':
            process_an_image = self.process_an_image_hist

        white_clip = clip1.fl_image(process_an_image)  # 对每一帧都执行lane_img_pipeline函数，函数返回的是操作后的image
        white_clip.write_videofile(self.outputfile, audio=False)


def main():
    if not os.path.isfile(args.inputfile):
        raise IOError('Error: --inputfile must refer to a video file, not directory.')

    demo = LaneDetection(args)
    demo.output()

if __name__ == "__main__":
    main()

