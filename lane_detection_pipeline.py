from utilities import *

def lane_detection_pipeline(img):

	copy_img = np.copy(img)
	# First step is to calibrate the camera
	ret, mtx, dist, rvecs, tvecs = calibrate_camera()

	#Next step is to undistort our images
	undistorted_img = undistort_image(copy_img, mtx, dist)
	copy_undistorted = np.copy(undistorted_img)

	#Next step is to Color/gradient threshold the image
	binary_image = gradient_color(copy_undistorted)
	# plt.imshow(binary_image,cmap='gray')
	# cv2.waitKey(500)
	# plt.show()

	#Next, a perspective transform is done on the image to have a top view.
	top_view_binary, M, Minv = perspective_transform(binary_image)
	# plt.imshow(top_view,cmap='gray')
	# cv2.waitKey(500)
	# plt.show()

	#Now that a top view is obtained, its time to detect the lanes
	out_img, left_fit, right_fit, left_fitx, right_fitx = fit_polynomial(top_view_binary)
	# plt.imshow(out_img)
	# plt.show()

	#Finally calculate curvature
	radius_of_curvature, center = measure_curvature(top_view_binary, left_fit, right_fit)


	# Write the curvature and center on the image
	undistorted_img = draw_text(undistorted_img, radius_of_curvature, 80, 80)
	undistorted_img = draw_text(undistorted_img, center, 80, 140)

	#plot everything together
	result = plot_lane_on_image(undistorted_img, top_view_binary, left_fitx, right_fitx, Minv)

	return result



# import os
# images = os.listdir("test_images/")

# for image in images:
# 	print("\nProcessing ",image,"\n")
# 	img = mpimg.imread('test_images/' + image)


# 	result = lane_detection_pipeline(img)
# 	# pts = np.array([[560,460],[715,460],[1150,720],[170,720]], np.int32)
# 	# pts = pts.reshape((-1,1,2))
# 	# img =cv2.polylines(img, [pts], True, (255,0,0), thickness=3)
# 	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# 	f.tight_layout()
# 	ax1.imshow(img)
# 	ax1.set_title('Original Image', fontsize=50)
# 	ax2.imshow(result)
# 	ax2.set_title('Detected Lane and Curvature', fontsize=50)
# 	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# 	plt.show()
# 	# exit(-1)
# 	#plt.savefig("test_images_output/"+image[:-4]+"DetectedLane.png")


#reading in a video
from moviepy.editor import VideoFileClip
from IPython.display import HTML

white_output = 'test_videos_output/output2_project_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/project_video.mp4")
white_clip = clip1.fl_image(lane_detection_pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))