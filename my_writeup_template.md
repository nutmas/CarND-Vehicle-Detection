# **Vehicle Detection**

---

###Vehicle Detection Project

**The goals / steps of this project are the following:**

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a colour transform and append binned colour features, as well as histograms of colour, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalise your features and randomise a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/binning_example.png
[image2a]: ./output_images/colour_space_YUV_car.png
[image2b]: ./output_images/colour_space_YUV_notcar.png
[image3a]: ./output_images/hog_car_image.png
[image3b]: ./output_images/hog_notcar_image.png
[image4]: ./output_images/feature_extraction.png
[image5]: ./output_images/window_search_area.png
[image6]: ./output_images/hot_windows_output.png
[image7a]: ./output_images/heatmap_1.png
[image7b]: ./output_images/heatmap_2.png
[image7c]: ./output_images/heatmap_3.png
[image7d]: ./output_images/heatmap_4.png
[image7e]: ./output_images/heatmap_5.png
[image7f]: ./output_images/heatmap_6.png
[video1]: ./output_project_video.mp4

---

### Files Submitted & Code Quality

#### 1. The project includes the following files:
* [VehicleDetector.ipynb](./VehicleDetector.ipynb) - notebook of vehicle detection pipeline.
* [svc_pickle.p](./svc_pickle.p) - pickled file containing trained model
* [output_project_video.mp4](./output_project_video.mp4) - video of annotated lane on project_video.mp4
* [writeup report](./my_writeup_template.md) - markdown file summarising the results

---


### Histogram of Oriented Gradients (HOG)

#### 1. Extracting HOG features from the training images.

In order to employ HOG feature detection I had to explore the images and utilise some functions prepare the images.

I started by reading in all the `vehicle` and `non-vehicle` images which can be seen in cell [2] of the python notebook.  
An example of each of the `vehicle` and `non-vehicle` classes is shown:

![alt text][image1]

To condition the image for HOG feature detection the following functions were created:
* Spatial Binning: Cell [4] of notebook: *this function takes in the image and resizes it from 64x64 pixels to 32x32 pixels and places the image into vector. The size of the image reduces by approximately 75%. This significant data size reduction will ease the loading on processing the image, but the image still has enough features to allow useful feature extraction.*

    Original Car Image size: (64, 64, 3)  Data type: float32  Features: 12288 
    Binned Car Image size:   (32, 32, 3)  Data type: float32  Features: 3072

* Colour Histogram: Cell [6] of notebook: using a histogram of each channel of the image some shape information can be extracted: *this function is fed an image in any colour space and produce a histogram of each channel. The following cell [7] allows visualisation of each colour space histogram when applied to the image. The purpose is to find where the car pixels can be detected and stand out compared with the notcar image.*

The image can be reduced to 96 features to identify it using the colour space histogram. 

``` 
Original Image size:     (64, 64, 3)  Data type: float32  Features: 12288
Colour Space Image size: (64, 64, 3)  Data type: float32  Features: 96
```

I explored the varying colour spaces and read some articles about this subject. With YCbCr colour space, colour structures can be recognised in a wide range of varying light conditions which are more compact than in other colour spaces. I thought the Y luminance component which appeared to show good results for grouping features for vehicle detections compared to non vehicle images. 

An example of the car and not car image histogram evaluation for YCbCr is shown:

![alt text][image2a]
![alt text][image2b]

* Hog Feature Detection: In cell [10] - *get_hog_features* is the function which takes in the image and set parameters and utilise the skimage feature detection function hog to compute the histogram of gradients from the image and return the features for an image.
    - In  this function a single channel or all the channel feature results can be used. I initially only selected Y channel hog detection as the features and trained models based on this. However on subsequent model builds I tried using all 3 hog channels and this resulted in better detection performance of the vehicles. I decided that 'ALL' channels setting of the hog as the parameter setting.

Examples of each channel of hog for YCbCr of vehicle and non-vehicle are shown below:

![alt text][image3a]
![alt text][image3b]
    
* Extract all features - the function in cell [16] 'extract_detail()' is created in order to combine all of the feature detections into one vector to supply the classifier. It calls the spatial binning, colour histogram and hog  features function to get the features from each of those processes. It takes the parameters set by me in order to influence the output each of the functions provide. Parameters set are:
    - input image: this was a YCbCr image reason for selecting this is already discussed.
    - Spatial binning size: after experimenting with 32 x 32, I tried 16 x 16 which di not give any noticeable negative performance and possibly improved the trained model. Reducing to 16 x 16 will reduce the size of image and therefore processing loading.
    - orient: I started with the default 9 as utilised in the course, and experimenting with different values found that 8 gave a good result for training a model.
    - HOG Channel selection was already discussed and the final choice was all 3 channels.
    - bins range - this parameter was selected as 0.0, 1.0 as I was importing png images which is not the same scale as an 8 bit image. I discovered later that this is a critical setting and needs to match with the image import, otherwise the feature detections are incorrect and produce some very poor results in training and detection.
    - in cell [18] I created a parameter array to hold these values so they could be easily saved and retrieved throughout the pipeline.

The final total number of features extracted from an image using binning, colour and hog features was:

`Total Number of features: 5568`


#### 3. Training a classifier using the selected features.

Before moving on to train a classifier  first had to prepare the dataset into training and testing sets. In cell [24] and [25] the data is prepared in the following way:
* Extract features using hog, colour and binning functions for both cars and notcars datasets.
* Create an array of the the feature vectors, with labels to identify each image as 'car' or 'notcar'
* To split the data I used the function from sklearn to split the data into 80:20 for training:test data. A random seed is first created to randomise the split each time it is run.
* StandardScaler is applied to training dataset to compute mean and standard deviation and be able to apply the same transform to the testing dataset.

The image (left to right) below shows:
* features of image
* Normalised features after applying Standard Scaler
* Reference visual of image

![alt text][image4]

In cell [27] I trained a linear SVC Classifier. Passing in the training and test sets the function trains a model then utilises the test data to predict the accuracy of the trained model, and then saves the model, scaler transform and parameters used for this model into a pickled file.
This file can be recalled in future to utilise the trained model to classify vehicles in an image.

The results of the training of the model are shown below:

```
11.82  Seconds to train SVC.
Test Accuracy of SVC =  0.9856
Trained SVC model predicts:  [1. 0. 1. 1. 1. 1. 1. 1. 0. 0.]
For these 10 labels:         [0. 0. 1. 1. 1. 1. 1. 1. 0. 0.]
0.00085 Seconds to predict: 10 labels with SVC
```

### Sliding Window Search

#### 1. Implementation of sliding window search.

I started implementing a sliding window search using the method from the course. This function is contained in cell [31]. I tried various parameter settings for widow size and overlap. I also set the 'start, stop' parameters to limit the search area. Numerous attempts and variations on the test images would sometimes get a large number of detections on some images, and then many false positives on others. The number of overlapping and size would also impact negatively on the speed performance.

To optimise the window coverage I created the function 'multi_windows()' in cell [32]. I created 4 sizes of window and positioned them at differing heights in the image. The smallest would be close to the horizon line. I added a horizontal overlap. As the images originally trained were 64x64 I used this size for the window size. The other 4 sizes were scaled from 64x64 with 1.5x, 2x and 2.5x to give me the range of sizes to cover the vehicles at different perspective points in the image. I aligned the top of each size of window at approximately top of the smallest box as the perspective angle this was the roof of the vehicle wherever it was in the image.

This worked very well and generated a lot less windows. The results in all images was improved and more consistent. An example of all the possible windows in the images is shown below:

![alt text][image5]

The windows positions determine where a vehicle could be detected in an image. The previous trained model is then used to search through each of these windows to return a prediction if there is a car present in the window or not. The function which utilises the model and generates the resulting filtered widows is contained in cell [34] 'search_for_cars()'.

The result is shown in the image:

![alt text][image6]

There are sometimes multiple windows representing each vehicle, but the false positives are minimal, and no detections on the empty image.

In order to reduce the multiple detections into a single window and reduce the false positives I implemented heatmap functions contained in cells [36],[37],[38] and [39]. 

These functions create a zero value image matching the size of the image the detection is being performed on. The functions iterate through all of the window boxes that have been highlighted as probably containing a vehicle if the pixel value is greater than the threshold set, then a non zero value is set for that pixel. The collections of non zero pixels then have a bounding box drawn around them, which results in one bounding box.

The images below:
Left: Resulting single bounding box encompassing the vehicles.
Right: The heatmap image that was generated.

![alt text][image7a]
![alt text][image7b]
![alt text][image7c]
![alt text][image7d]
![alt text][image7e]
![alt text][image7f]

This concludes the pipeline for single image detection, and the results can be seen in the images above, the detection and classification of vehicles in an image by drawing bounding boxes around the vehicle. 

---

### Video Implementation

#### Utilising the same trained model and single image functions the vehicle detection is applied to the video.

In cell[41] 'processImage( )' - a function is created to utilise the same pipeline and functions used for single image detection and apply it to video.

In order to stabilise the detection and reduce the impact of false positives, a history tracker is created in the video pipeline. This function appends the heatmaps at each cycle, and averages the sum. Once the stack exceeds the set threshold it will pop off the old windows and add the new detections.
This function can be seen in lines 28-38 of cell[41].

The video output can be seen at: [link to video result](output_project_video.mp4)

The vehicles in the video are successfully detected and tracked, with minimal noise and false positives.

---

### Discussion


#### 1. Discussion of problems with my implementation of this project. 

I found 2 areas that greatly influenced the performance of this project; data preparation and window positions & size parameters.  

* After finding a lot of false positives in the single image processing, I investigated the data images in detail. I found many examples of full and partial vehicle images in the non-vehicle dataset. This was a manual process that took a long time to work through. As I found images that contained vehicles or enough of a partial vehicle to cause conflict; I deleted the images from the dataset. I did not find all instances as it was quite time consuming and had to randomly check batches to speed up my manual search. The result of this activity did improve the score of the classifier and at one point I had just slightly over 99% score for classification.  
* The window sizes, overlapping and positions can greatly influence the performance of detection. I think there is still a lot of improvement opportunity to get improve the positions and overlaps, therefore obtaining a good balance between processing speed and useful detections.

The limitation of my pipeline is it is only suitable for the position of the car in the images and video. If the position of the car were to move to the centre lane then the vehicle detection would not be fully present on the left lane; so a blindspot would exist. The vehicle is also fixed to this camera perspective and will deteriorate if the horizon line were to fluctuate greatly.

