# ML-project2-Vehicle-Detection


## Overview:
Vehicle detection is a important part of self-driving car technology. Usually in a self-driving car, camera is the most common sensor, from which if we can detect the vehicles around us, we can plan trajectory and avoid collisions. So in this project, we will implement a system, which the input is a image from a on-vehicle camera and the output looks like following:
<div align=center><img width="800"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/figure-1.png"/></div>

## Motivation:
In this project, we used the YOLO network to detect vehicles. This is a different method comparing with traditional vehicle detection algorithm. In the latest version, the detection speed on the GPU can basically meet the real-time-detection's requirement. The YOLO network is so important and unique, so that we chose to implement it in this project, in order to learn the technical details of YOLO, and to practice our ability to implement and train neural networks.


## Dataset:
In this project, we will use the [BDD100K dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/), which includes 100,000 images of size (1280 * 720) pixels.
Here is an example:

<div align=center><img width="600"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/ac9be3fe-790d1f8e.jpg"/></div>

Besides, there is also a label file of json, which we can use to find the ground truth detection-boxs' information:
```
[
   {
      "name": str,
      "timestamp": 1000,
      "category": str,
      "bbox": [x1, y1, x2, y2],
      "score": float
   }
]
```
 According to [dataset info](https://github.com/ucbdrive/bdd-data): Box coordinates are integers measured from the top left image corner (and are 0-indexed). [x1, y1] is the top left corner of the bounding box and [x2, y2] the lower right. name is the video name that the frame is extracted from. It composes of two 8-character identifiers connected '-', such as c993615f-350c682c. Candidates for category are ['bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider']. In the current data, all the image timestamps are 1000. In our case, we will use only the attributes "bbox" and "category".
 
In our actual training process, we found that many of the marked objects were too detailed, which caused that there are too many labels overlaped with each other, and it also has higher requirements to train network, so we preprocessed the dataset before training: for each category we only saved 5 objects with the largest bbox area.
 
## Method:
The main idea is to use pre-trained neraul network called [YOLO network](https://pjreddie.com/darknet/yolo/) as the basic model, and we will try to retrain it in order to make it more suitable for our task. As a result, we will apply this detection model into images but also videos. And then we implement a traditional object detector based on SVM, in order to compare with YOLO.

## Plan:
We divided the whole project into 3 part: data process, training YOLO model, evaluate the model and apply into videos.

| Name | Work |
|:----:|:------:|
|Xi  | data process |
|Martin | YOLO model |
|Ziyuan | apply model|

## YOLO network
YOLO's name comes from "you only look once", which exactly explained the mian idea of YOLO network: it reads in a image, predicts the area of the object in the image and also the category of object in this area, because it only needs to read in the image and go through the neural network at a time. the speed of detection processing could be very fast.

### Architecture:
Its architecture is as follows:

![](https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/YOLO-archi.png)


In our implementation, the structure is shown as the following table:

| Layer | Details |
|:----:|:------:|
|Inception model (first 20 layers)  | well pre-trianed layers, to extract features, output size = {6 * 6} |
|Convolutional layer | filter size = {} |
|Convolutional layer | filter size = {} |
|Convolutional layer | filter size = {} |
|Convolutional layer | filter size = {} |
|Dense layer | size = {}|
|Dense layer | size = {}|
|output layer | output size = {}|

Finally we can resize the output of NN into a 3D tensor: grid size * grid size* ( class amount + anchor box amout * 5 ), in our case: 15 * 15 * (10 + 2 * 5), shown as following ( source: [deepsystem.io](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_1509) ): 

![](https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/nn-output.PNG)

The loss function is shown as follows, in fact, the main idea is to convert object-detection into a regression problem:

<div align=center><img width="600"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/loss.png"/></div>

### Training process
During training process, we found that training such a large neural network is very time consuming, and the network is very easy to overfit. After trying many different methods, including image augumentation, adding dropout. Still overfitting:

![](https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/overfiting.jpg)

So we used the trained model: darknet provided by author, which got impressive results:

<div align=center><img width="600"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/predictions.jpg"/></div>

### Compare with traditinal detection model
Besides YOLO model, we got a traditinal detection model based on SVM. We use a [two-classes-dataset](http://www.gti.ssr.upm.es/data/Vehicle_database.html) to train SVM to classify {"car","non-car"}.
The main workflow is shown as following:

>1. extract feature and train svm model on training data.
>2. pick test image and use windows of different sizes to slide.
>3. resize this window-images and classify with well-trained SVM, label the "car" box.
>4. reducing redundant box with non-maximum-suppression

In details, we using HOG( Histogram of oriented gradient ) feature from HUV channels, which are highly frequently used by many image classification problem: 

<div align=center><img width="800"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/hog.png"/></div>

And here is an example:

<div align=center><img width="400"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/car-notcar.png"/></div>

<div align=center><img width="400"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/hog-SVM.png"/></div>


As we can see, an important step of detector with SVM is to use different size windows to slide, which actually needs many times classification for every single image, and it seriously reduces the detection efficiency. Besides, a huge disadvantage of this SVM-based detector is the poor generalization ability, which may be related to the features we choose.

Compare the classification on a clear images:

from SVM detector:
<div align=center><img width="600"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/output-bboxes-svm.png"/></div>

and YOLO detector:









