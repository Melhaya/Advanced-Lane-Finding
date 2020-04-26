import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_camera():

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (9,6), corners, ret)

    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def adjust_gamma(image, gamma=0.7):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def undistort_image(img, mtx, dist):

    return cv2.undistort(img, mtx, dist, None, mtx)

def gradient_color(img, s_thresh=(170, 255), l_thresh=(220, 255), b_thresh=(190,255), sx_thresh=(20, 100)):
    
    HLS_img = np.copy(img)
    LAB_img = np.copy(img)
    Gray_image = np.copy(img)

    # Convert to HLS color space and separate the channels
    hls = cv2.cvtColor(HLS_img, cv2.COLOR_RGB2HLS)
    l_HLSchannel = hls[:,:,1]
    s_HLSchannel = hls[:,:,2]

    s_HLSchannel=s_HLSchannel*(255/np.max(s_HLSchannel))
    l_HLSchannel=l_HLSchannel*(255/np.max(l_HLSchannel))
    
    # Threshold color channel
    s_binary = np.zeros_like(s_HLSchannel)
    s_binary[(s_HLSchannel >= s_thresh[0]) & (s_HLSchannel <= s_thresh[1])] = 1
    
    l_HLSbinary = np.zeros_like(l_HLSchannel)
    l_HLSbinary[(l_HLSchannel > l_thresh[0]) & (l_HLSchannel <= l_thresh[1])] = 1
                        ####################################
    # Convert to LAB color space and separate the channels
    LAB = cv2.cvtColor(LAB_img, cv2.COLOR_RGB2LAB)
    l_LABchannel = LAB[:,:,0]
    b_LABchannel = LAB[:,:,2]
    
    if np.max(b_LABchannel) > 175:
        b_LABchannel = b_LABchannel*(255/np.max(b_LABchannel))

    b_LABbinary = np.zeros_like(b_LABchannel)
    b_LABbinary[((b_LABchannel > b_thresh[0]) & (b_LABchannel <= b_thresh[1]))] = 1
                        ####################################

    gray = cv2.cvtColor(Gray_image, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
                        ####################################

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(b_LABbinary)
    combined_binary[(l_HLSbinary == 1) | (b_LABbinary == 1)] = 1

    return combined_binary

def perspective_transform(img):

    img_size =(img.shape[1], img.shape[0])
    #define 4 source points src = np.float32([[,],[,],[,],[,]])
    src = np.float32([[560,460],[715,460],[1150,720],[170,720]])         
    #define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    offset = 100
    dst = np.float32([[offset, 0],
                     [img_size[0]-offset, 0], 
                     [img_size[0]-offset, img_size[1]], 
                     [offset, img_size[1]]])      

    #use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv =  cv2.getPerspectiveTransform(dst, src)
    #use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv



def find_lane_pixels(top_view_binary):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(top_view_binary[top_view_binary.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((top_view_binary, top_view_binary, top_view_binary))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 10
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(top_view_binary.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = top_view_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = top_view_binary.shape[0] - (window+1)*window_height
        win_y_high = top_view_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(top_view_binary):

    #binary_warped = np.copy(top_view)

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(top_view_binary)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, top_view_binary.shape[0]-1, top_view_binary.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, left_fitx, right_fitx

def measure_curvature(top_view_binary, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    ploty = np.linspace(0, top_view_binary.shape[0]-1, top_view_binary.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3/100 #30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/378 #3.7/700 # meters per pixel in x dimension
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curvature =  ((1 + (2*left_fit_cr[0] *y_eval*ym_per_pix + left_fit_cr[1])**2) **1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Calculate vehicle center
    #left_lane and right lane bottom in pixels
    left_lane_bottom = (left_fit[0]*y_eval)**2 + left_fit[0]*y_eval + left_fit[2]
    right_lane_bottom = (right_fit[0]*y_eval)**2 + right_fit[0]*y_eval + right_fit[2]
    
    # Lane center as mid of left and right lane bottom                        
    lane_center = (left_lane_bottom + right_lane_bottom)/2.
    center_image = 640
    center = (lane_center - center_image)*xm_per_pix #Convert to meters
    position = "left" if center < 0 else "right"
    center = "Vehicle is {:.2f}m {}".format(center, position)
    radius = 'Radius of curvature: {} m'.format(int(np.average([left_curvature, right_curvature]))) 

    # Now our radius of curvature is in meters
    return radius, center



def draw_text(img, text, x, y):
        return cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

def plot_lane_on_image(undistorted_img, top_view_binary, left_fitx, right_fitx, Minv):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(top_view_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, top_view_binary.shape[0]-1, top_view_binary.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (top_view_binary.shape[1], top_view_binary.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    # plt.show()
    return result

