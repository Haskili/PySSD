<h1 align="center">PySSD</h1> 
  <p align="center">
    A Tensorflow Single-Shot-Detector implementation
    <br/><br/>
    [<a href="https://www.tensorflow.org/">Tensorflow</a>]
    [<a href="https://github.com/Haskili/PySSD#acknowledgements">Acknowledgements</a>]
    [<a href="https://github.com/Haskili/PySSD/issues">Issues</a>]
    <br/><br/><br/>
    <img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/1200px-TensorFlowLogo.svg.png" alt ="" width="10%" height="10%">
    <img src = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRLgwjF2JFQ-jvZn53x4bIhemYv7dLQKROIVg&usqp=CAU" alt ="" width="20%" height="20%">
  </p>
</p>

## Overview

PySSD is an implementation of a [Single-Shot-Detector](https://arxiv.org/abs/1512.02325) (SSD) in Tensorflow, and functions as a basic library as well. 

An SSD is a multibox detector that classifies objects within an image. In my previous projects I have experienced working with both SSD's and You-Only-Look-Once (YOLO) using the OpenCV 4.X library and utilizing pretrained models/configurations. However, this is a much more pure and basic implementation of something that would be more functionally more portable.

<br>
<p align="center"><img src = "https://github.com/Haskili/PySSD/blob/main/Images/example_result.png" alt ="" width="30%" height="30%"></p>
<br>

It is meant to be used as either a standalone program to experiment with or included as part of a larger workflow. Included are all the supporting functions required to implement it into any project with relative ease, such as labelmap parsing and modular output-access methods that clearly depict the way the results are being handled.

<br></br>

## Requirements

**Model & Configuration Files**

To use this library, it is required that you utilize a trained SSD model.<br>
(e.g. https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)

This library utilizes the `savedModel` format, but depending on the model you choose, other things such as how certain values of the output are accessed may change as well. By default, you are able to alter how the values are parsed in the returned output from a forward pass using the aforementioned model by changing the following constants defined in `__int__.py` of the `PySSD` module:
```Python
# Set the variables for indexing into the 
# return data-structure of the model forward-pass
NET_FORMAT_NUMDET  = 'num_detections'
NET_FORMAT_BOXES   = 'detection_boxes'
NET_FORMAT_CLASSES = 'detection_classes'
NET_FORMAT_SCORES  = 'detection_scores'
```
<br>
Additionally, the model will likely also require a labelmap that can be parsed with the tools included in the module.
<br>
For the aforementioned model (which is trained on COCO images), you can use the COCO labelmap available in many repositories online.

To specify the labelmap file, set the following constant in the `__init__.py` of the `PySSD` module:
```Python
LMF_PATH = 'mscoco_label_map.pbtxt'
```
<br></br>

## Dependencies

**Tensorflow (2.6.0-3)**
```sh
pacman -S python-tensorflow
```
Please see dependencies listed [here](https://aur.archlinux.org/packages/python-tensorflow-git/).

<br>

**NumPy (1.20.3-1)**
```sh
pacman -S python-numpy
```
Please see dependencies listed [here](https://archlinux.org/packages/extra/x86_64/python-numpy/).

<br>

**Pillow (8.3.1-1)**
```sh
pacman -S python-pillow
```
Please see dependencies listed [here](https://archlinux.org/packages/community/x86_64/python-pillow/).

<br></br>

## Usage

To run the program, all that's needed is to call the driver file, `SSD.py`
```sh
./python SSD.py
```
<br>

To configure the driver to run a custom set of settings, edit the following in the `__init__.py` of the `PySSD` module:
```Python
# Set the values for indexing in the label map if it's 
# a 'pbtxt', which gets returned as a nested dictionary
LMF_FORMAT_ID    = 'id'
LMF_FORMAT_CLASS = 'display_name'
LMF_FORMAT_ALT   = 'name'

# Set the variables for indexing into the 
# return data-structure of the model forward-pass
NET_FORMAT_NUMDET  = 'num_detections'
NET_FORMAT_BOXES   = 'detection_boxes'
NET_FORMAT_CLASSES = 'detection_classes'
NET_FORMAT_SCORES  = 'detection_scores'

# Set a confidence value for filtering detection results
CONF_THRESHOLD = 0.60

# Define the basic user-input values
IMG_PATH = './Images/DogPark3.jpg'
NET_PATH = './'
LMF_PATH = 'mscoco_label_map.pbtxt'
```
<br>

**Example**
```sh
python3 SSD.py 
Generating label map from "mscoco_label_map.pbtxt"
Loading model at "./"
Running input "./Images/DogPark3.jpg" through the model

Detection Class: 1 -- person
Detection Score: 85.793%
       Location: (2212, 354) -- (3073, 1922)

Detection Class: 18 -- dog
Detection Score: 82.915%
       Location: (24, 1169) -- (1216, 2039)

Detection Class: 18 -- dog
Detection Score: 74.295%
       Location: (1543, 958) -- (2222, 2370)

Finished with input "./Images/DogPark3.jpg"
```

<br></br>

## Acknowledgements
*'PySSD'*  is my attempt at creating a simple implementation of a portable Single-Shot-Detector library based on the model(s) found on the official Tensorflow repository. Additionally, it it also meant to serve as a learning tool for SSD's and what a typical implementation might look like. Many if not most of the commenting in the library is meant to be informative, and is liable to be overly explicit.
