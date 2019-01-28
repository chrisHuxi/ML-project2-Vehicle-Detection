# ML-project2-Vehicle-Detection


## Overview:
Vehicle detection is a important part of self-driving car technology. Usually in a self-driving car, camera is the most common sensor, from which if we can detect the vehicles around us, we can plan trajectory and avoid collisions. So in this project, we will implement a system, which the input is a image from a on-vehicle camera and the output looks like following:
<div align=center><img width="700"  src="https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/figure_1.png"/></div>


## Dataset:
In this project, we will use the [BDD100K dataset](https://bair.berkeley.edu/blog/2018/05/30/bdd/), which includes 100,000 images of size (1280 * 720) pixels.
Here is an example:

![](https://github.com/chrisHuxi/ML-project2-Vehicle-Detection/blob/master/readme_img/ac9be3fe-790d1f8e.jpg)

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


## Method:
The main idea is to use pre-trained neraul network called [YOLO network](https://pjreddie.com/darknet/yolo/) as the basic model, and we will try to retrain it in order to make it more suitable for our task. As a result, we will apply this detection model into images but also videos.

## Plan:
We divided the whole project into 3 part: data process, training YOLO model, evaluate the model and apply into videos.

| Name | Work |
|:----:|:------:|
|Xi  | data process |
|Martin | YOLO model |
|Ziyuan | apply model|
