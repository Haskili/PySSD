<h1 align="center">PySSD</h1> 
  <p align="center">
    A Tensorflow Single-Shot-Detector implementation
    <br/><br/><br/>
    [<a href="https://www.tensorflow.org/">Tensorflow</a>]
    [<a href="https://github.com/Haskili/PySSD#acknowledgements">Acknowledgements</a>]
    [<a href="https://github.com/Haskili/PySSD/issues">Issues</a>]
  </p>
</p>

## Overview

...
<br></br>

## Requirements

**Tensorflow (2.X)**
```
pacman -S python-tensorflow
```
Please see dependencies listed [here](https://aur.archlinux.org/packages/python-tensorflow-git/).
<br></br>

**Model Files**
<br></br>
...
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
