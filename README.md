**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images_output/straight_lines2_undistorted.png "Undistorted"
[image2]: ./test_images_output/straight_lines2_Binary_image.png "Road Binary Example"
[image3]: ./test_images_output/straight_lines2Topview.png "Original Vs Topview Example"
[image4]: ./test_images_output/straight_lines2Topview_binary.png "Topview Vs topview binary Example"
[image5]: ./test_images_output/straight_lines2Topview_polynomial.png "Fitted Polynomial"
[image6]: ./test_images_output/straight_lines2DetectedLane.png "Output"
[video1]: ./test_videos_output/output_project_video.mp4 "Video"
[video2]: ./test_videos_output/harder_challenge_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in my `utilities.py`, `calibrate_camera()` function.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function which is found in `utilities.py`, `undistort_image()` function.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image in `gradient_color()` funcrion in `utilities.py`.  Here's an example of my output for this step.
![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in `utilities.py`. The `perspective_transform()` function takes as inputs an image (`img`). I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32([[560,460],
                  [715,460],
                  [1150,720],
                  [170,720]])

offset = 100
dst = np.float32([[offset, 0],
                  [img_size[0]-offset, 0], 
                  [img_size[0]-offset, img_size[1]], 
                  [offset, img_size[1]]]) 
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 460      | 100, 0        | 
| 715, 460      | 1180, 0      |
| 1150, 720     | 1180, 720      |
| 170, 720      | 100, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then, the `fit_polynomial()` function is called to detect the lanes and fit a polynomial on their positions. A 2nd order polynomial was fitted on then and the results appeared as follows:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the `measure_curvature()` function in my code in `utilities.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `plot_lane_on_image()` function in my code in `utilities.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_videos_output/ 	output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As seen above, the pipeline implented worked well with the project video. However, when I inputted the harder challenge video, the results were really bad. the pipeline was unable to detect the lanes accuratley. One of the challenges I noticed in the video was the brightness of the sun. It affected the pipeline and thus causing this wrong detection. Perhaps there are more filters that can be used to combat the difference in brightness and shadow between the lanes.
