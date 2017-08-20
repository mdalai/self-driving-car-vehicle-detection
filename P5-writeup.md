# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[example]: ./assets/example.PNG
[image1]: ./assets/resized_car_image.PNG
[image2]: ./assets/color_histogram.PNG
[his_error]: ./assets/color_histogram_error.PNG
[image3]: ./assets/hog41.PNG
[image20]: ./assets/hog20.PNG
[image21]: ./assets/hog21.PNG
[image22]: ./assets/hog22.PNG
[image_norm]: ./assets/normalization2.PNG
[image_preds]: ./assets/prediction.PNG
[image_slide1]: ./assets/slide_window1.PNG
[image_slide2]: ./assets/slide_window2.PNG
[image_slide21]: ./assets/slide_window21.PNG
[image41]: ./assets/pipeline_test1.PNG
[image42]: ./assets/pipeline_test_all.PNG
[image51]: ./assets/heatmap1.PNG
[image52]: ./assets/heatmap2.PNG
[image61]: ./assets/heatmap61.PNG
[image62]: ./assets/heatmap62.PNG

[image7]: ./assets/pipeline_test_all2.PNG


[video1]: ./project_video_output3.mp4

_[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points: 
Here I will consider the rubric points individually and describe how I addressed each point in my implementation._  

---
## Data Exploration
- Load all the `vehicle` and `non-vehicle` images. All png files.
- A count of 8792  cars and 8968  non-cars
- Size of each image:  (64, 64, 3)

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][example]

## Feature Extraction
### Spatial features (raw pixel features) 
- Compute Spatial features: extract raw pixel features. 
- Resize the image: We do not want all pixels because it is computational expensive. Therefore we resize the image into smaller size. 
- Transform into feature vector by RAVEL().

![alt text][image1]

### Color histogram features
- Compute Color histogram features: cars can be identified by its color. 
- Different color space can be adapted: RGB, HSV ...

It has problem if using Matplotlib image to read the image. Matplotlib image reads PNG file in scale of 0 to 1. See the example:

![alt text][his_error]

Then I used OpenCV to read instead. Much better results, see bellow image:

![alt text][image2]

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][image3]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image20]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters:
- `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
- `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`
- `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(4, 4)`

It seems middle one works best. And each channel HOG works to differentiate Car and NotCar. So we should combine all Channels as final HOG features.

![alt text][image20]

![alt text][image21]

![alt text][image22]

### Combine all features and normalization
- Combine: we will combine all features extracted from above steps.
- Normalization: We want all features contribute evenly to the model training. We will use Scikit-learn StandardScaler. Here is an example of normalization:

![alt text][image_norm]


## Train Classifier
#### Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using:
- Combination of spacial features, color histogram features and HOG features.
- Normalization and Shuffle, Split with Scikit-learn handy tools.

I tried 6 times with different parameters. The final one I decide to use:
```
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```
I got following training results:
```
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
21.73 Seconds to train SVC...
Test Accuracy of SVC =  0.9932
```

Here is an example of predictions using trained SVM:

![alt text][image_preds]

**Test Result:**
- Cars predictions are all corrrect.
- NotCars predictions 3 corrrect 7 incorrect.
- **Conclusion**: 
  - The model struggles in predicting notCar images. It seems that it predicts images with line shape as car. 
  - Logically it is better to have false positive than false negative in driving scenario. False negative means it is car but predict as NotCar. Driver may end up hitting a vehicle in front.
  - Of course, we have to find a solution to eliminate false positive as well. 

## Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image_slide1]

I applied following techniques:
- Interest areas: ystart = 400, ystop = 656
- Sliding window is too small, need to scale.

Two ways to scale: scale the whole image smaller OR scale the sliding window bigger.

scale the whole image smaller: scale = 1.5; the result:

![alt text][image_slide2]

Scale up the sliding window size: scale = 1.5;  the window size will (96,96). The result:

![alt text][image_slide21]

I decided to use later one because the first one did not work out for me. To use later one, we have to resize window size to (64,64) before applying classifier.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image41]

![alt text][image42]
---

## Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output3.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here is an example of applying heatmap:

![alt text][image51]

**The results is NOT Satisfying!!! In order to apply heatmap, we need to have many bounding boxes detection. It is too few now. I need more overlapping detections. I updates the overlapping parameter and try it again**

![alt text][image52]

**Much better!!!**

I added the heatmap technique into the pipeline. Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


I tried following heatmap thresholds: 1, 0.5, 0.75, 0.85, 0.95. Best: **heat_threshold = 0.75**.

### Here are six frames and their corresponding heatmaps, resulting bounding boxes:

![alt text][image62]


### Create final pipeline and test results:

```
##################### PARAMETERS ####################################################################
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
scale = 1.5 # scale the sliding window size
overlap = 0.75 # 0.5 # overlapping parameter for slide window
heat_threshold = 0.75 # heatmap threshold
```

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

- Matplotlib image .PNG file reading pixel values [0,1]. This screws up all my works in the first part. I just couldn't find the problem why my classifier works so poor. DO NOT Recommend convert [0,1] to [0, 255] and then scale it. Just keep the [0,1] and train the classifier. When reading the image frame, remember to divide by 255. 
- TOO SlOW applying pipeline into the video. I think following techniques can improve the processing speed:
  - Incease pix_per_cell, cell_per_block from 8,2 to 16,4. 
  - Appy HOG feature extraction on the whole image once.
  - Decrease spacial color size. (32,32) --> (16,16). It will decrease the feature amount. So that the model can process faster.
- Still have many false positives, may apply:
  - apply different sliding window sizes: (64,64), scale 1.2, scale 1.5.
  - Keep tunning parameters when training the classifier. Find the best parameters by Sklearn GridSearchCV tool.
  - Apply more training data to train the classifier. We need a better classifier.
  - Try deep learning classifier which has CNN may work better.

