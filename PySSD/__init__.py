# Base libraries
import os
import pathlib
import re

# 'Tensorflow Binary Fix'
# 
# This is a workaround for any issues 
# regarding the TF binary being built 
# for using SSE3, AVX, etc.
#
# Aditionally, you could also use
# 'export TF_CPP_MIN_LOG_LEVEL = 2'
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Numpy
import numpy as np

# Tensorflow
import tensorflow as tf

# Image processing
from PIL import Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

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

def parse_detections(image, confidence_threshold, detections, labels):
    """
    Parse the detections dictionary for each detection and evaluate
    each detection 

    Args:
        confidence_threshold: Threshold to filter out uncertain detections
        detections: Dictionary of detection results
        labels: Dictionary of the parsed labelmap file

    Returns:
        None

    Raises:
        N/A
    """

    # Iterate through each possible detection
    for idx in range(0, detections[NET_FORMAT_NUMDET]):

        # Get the class label index, the string label at 
        # that index, and the classification confidence 
        # value for the current detected object
        classification = detections[NET_FORMAT_CLASSES][idx]
        confidence     = detections[NET_FORMAT_SCORES][idx]
        class_label    = labels[str(classification)][LMF_FORMAT_CLASS]

        # Check to make sure the detection confidence value
        # meets or exceedes the given threshold
        if confidence < confidence_threshold:
            continue

        # Calculate the detected object's 
        # normalized coordinates
        x_min = round(detections[NET_FORMAT_BOXES][idx][1]*image.width)
        y_min = round(detections[NET_FORMAT_BOXES][idx][0]*image.height)
        x_max = round(detections[NET_FORMAT_BOXES][idx][3]*image.width)
        y_max = round(detections[NET_FORMAT_BOXES][idx][2]*image.height)

        # Draw a bounding box on the image indicating 
        # the detected object location
        drawing = ImageDraw.Draw(image)
        drawing.line(
            [
                (x_min, y_min), 
                (x_min, y_max), 
                (x_max, y_max), 
                (x_max, y_min), 
                (x_min, y_min)
            ],
            width = 3,
            fill  = 'green'
        )

        # Write the classification on top of the bounding box
        drawing.text(
            (x_min+(0.03*image.width), y_min-(0.03*image.height)),
            f"{class_label}: {round(confidence*100.0, 3)}",
            (0, 255, 0), 
            ImageFont.load_default()
        )

        # Print the detection results
        print(f"Detection Class: {classification} -- {class_label}")
        print(f"Detection Score: {round(confidence * 100.0, 3)}%")
        print(f"       Location: ({x_min}, {y_min}) -- ({x_max}, {y_max})\n")

    return

def forward_pass(model, image):
    """
    Run a forward pass of an input image through
    the given model and return back the parsed 
    results

    Args:
        model: A model (tf.saved_model format) to run data through
        image: Image input data to pass through a model

    Returns:
        Parsed tensor of maps (dependant on model used)

    Raises:
        N/A
    """

    # Load in the image and then convert it
    image = np.array(image, dtype = 'uint8')
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Give the input tensor (image) to the model and save the output
    output = model(input_tensor)

    # Parse the output dictionary returned from the model's forward pass;
    # This includes recreating it from the keys of the current output dictionary
    #
    # e.g. dict_keys(['detection_boxes', 'detection_scores', etc.]) 
    #      where each key corrosponds to an 2-3 dimensional array of 
    #      values that refer to the same detection at that particular 
    #      index in all other key's arrays 
    #
    num_detections = int(output.pop('num_detections'))
    output = {key:val[0, : num_detections].numpy() for key, val in output.items()} 
    output['num_detections'] = num_detections
    output['detection_classes'] = output['detection_classes'].astype(np.int64)
    return output

def get_LM_PBTXT(fname, id_idx):
    """
    Parses the specified .pbtxt file 'fname' to get 
    a nested dictionary of all the different classes 
    indexed by their ID value 'id_idx'

    Args:
        fname: File name of label map data
        id_idx: The 'variable' name of the human-readable class names
                (e.g. display_name: "motorcycle")

    Returns:
        A dictionary formatted as follows,
            ('key': Class ID Number, 'value': Class Information}

    Raises:
        N/A
    """

    file_text = " "
    with open(fname, 'r') as reader:
        file_text = reader.read()

    # Find each 'item' instance describing a class
    re_main = re.findall('.+\{\n[^\}]+\}', file_text)

    # Iterate through each 'item' instance found
    label_map = {}
    for item in re_main:

        # Find all 'attributes' of the current 'class'
        class_dict = {}
        re_attr = re.findall('[^\ ]+:.*\n', item)

        # Iterate through all 'attributes' of the current 'item'
        for attr in re_attr:

            # Append all 'attributes' and their values to
            # the dictionary of the current 'item'
            attribute = attr.split(':')
            class_dict.update({attribute[0]:attribute[1][1:-1].strip("'\"")})

        label_map.update({class_dict[id_idx]:class_dict})

    return label_map

def get_LM_STATIC():
    """
    Returns a static dictionary of classes mapped along
    with their class ID number

    Args:
        N/A

    Returns:
        A dictionary formatted as follows,
            ('key': Class ID Number, 'value': Class Information}

    Raises:
        N/A
    """

    labels = {1:   {'id': '1', 'display_name': 'unlabeled'},
              2:   {'id': '2', 'display_name': 'person'},
              3:   {'id': '3', 'display_name': 'bicycle'},
              4:   {'id': '4', 'display_name': 'car'},
              5:   {'id': '5', 'display_name': 'motorcycle'},
              6:   {'id': '6', 'display_name': 'airplane'},
              7:   {'id': '7', 'display_name': 'bus'},
              8:   {'id': '8', 'display_name': 'train'},
              9:   {'id': '9', 'display_name': 'truck'},
              10:  {'id': '10', 'display_name': 'boat'},
              11:  {'id': '11', 'display_name': 'traffic'},
              12:  {'id': '12', 'display_name': 'fire'},
              13:  {'id': '13', 'display_name': 'street'},
              14:  {'id': '14', 'display_name': 'stop'},
              15:  {'id': '15', 'display_name': 'parking'},
              16:  {'id': '16', 'display_name': 'bench'},
              17:  {'id': '17', 'display_name': 'bird'},
              18:  {'id': '18', 'display_name': 'cat'},
              19:  {'id': '19', 'display_name': 'dog'},
              20:  {'id': '20', 'display_name': 'horse'},
              21:  {'id': '21', 'display_name': 'sheep'},
              22:  {'id': '22', 'display_name': 'cow'},
              23:  {'id': '23', 'display_name': 'elephant'},
              24:  {'id': '24', 'display_name': 'bear'},
              25:  {'id': '25', 'display_name': 'zebra'},
              26:  {'id': '26', 'display_name': 'giraffe'},
              27:  {'id': '27', 'display_name': 'hat'},
              28:  {'id': '28', 'display_name': 'backpack'},
              29:  {'id': '29', 'display_name': 'umbrella'},
              30:  {'id': '30', 'display_name': 'shoe'},
              31:  {'id': '31', 'display_name': 'eye'},
              32:  {'id': '32', 'display_name': 'handbag'},
              33:  {'id': '33', 'display_name': 'tie'},
              34:  {'id': '34', 'display_name': 'suitcase'},
              35:  {'id': '35', 'display_name': 'frisbee'},
              36:  {'id': '36', 'display_name': 'skis'},
              37:  {'id': '37', 'display_name': 'snowboard'},
              38:  {'id': '38', 'display_name': 'sports'},
              39:  {'id': '39', 'display_name': 'kite'},
              40:  {'id': '40', 'display_name': 'baseball'},
              41:  {'id': '41', 'display_name': 'baseball'},
              42:  {'id': '42', 'display_name': 'skateboard'},
              43:  {'id': '43', 'display_name': 'surfboard'},
              44:  {'id': '44', 'display_name': 'tennis'},
              45:  {'id': '45', 'display_name': 'bottle'},
              46:  {'id': '46', 'display_name': 'plate'},
              47:  {'id': '47', 'display_name': 'wine'},
              48:  {'id': '48', 'display_name': 'cup'},
              49:  {'id': '49', 'display_name': 'fork'},
              50:  {'id': '50', 'display_name': 'knife'},
              51:  {'id': '51', 'display_name': 'spoon'},
              52:  {'id': '52', 'display_name': 'bowl'},
              53:  {'id': '53', 'display_name': 'banana'},
              54:  {'id': '54', 'display_name': 'apple'},
              55:  {'id': '55', 'display_name': 'sandwich'},
              56:  {'id': '56', 'display_name': 'orange'},
              57:  {'id': '57', 'display_name': 'broccoli'},
              58:  {'id': '58', 'display_name': 'carrot'},
              59:  {'id': '59', 'display_name': 'hot'},
              60:  {'id': '60', 'display_name': 'pizza'},
              61:  {'id': '61', 'display_name': 'donut'},
              62:  {'id': '62', 'display_name': 'cake'},
              63:  {'id': '63', 'display_name': 'chair'},
              64:  {'id': '64', 'display_name': 'couch'},
              65:  {'id': '65', 'display_name': 'potted'},
              66:  {'id': '66', 'display_name': 'bed'},
              67:  {'id': '67', 'display_name': 'mirror'},
              68:  {'id': '68', 'display_name': 'dining'},
              69:  {'id': '69', 'display_name': 'window'},
              70:  {'id': '70', 'display_name': 'desk'},
              71:  {'id': '71', 'display_name': 'toilet'},
              72:  {'id': '72', 'display_name': 'door'},
              73:  {'id': '73', 'display_name': 'tv'},
              74:  {'id': '74', 'display_name': 'laptop'},
              75:  {'id': '75', 'display_name': 'mouse'},
              76:  {'id': '76', 'display_name': 'remote'},
              77:  {'id': '77', 'display_name': 'keyboard'},
              78:  {'id': '78', 'display_name': 'cell'},
              79:  {'id': '79', 'display_name': 'microwave'},
              80:  {'id': '80', 'display_name': 'oven'},
              81:  {'id': '81', 'display_name': 'toaster'},
              82:  {'id': '82', 'display_name': 'sink'},
              83:  {'id': '83', 'display_name': 'refrigerator'},
              84:  {'id': '84', 'display_name': 'blender'},
              85:  {'id': '85', 'display_name': 'book'},
              86:  {'id': '86', 'display_name': 'clock'},
              87:  {'id': '87', 'display_name': 'vase'},
              88:  {'id': '88', 'display_name': 'scissors'},
              89:  {'id': '89', 'display_name': 'teddy'},
              90:  {'id': '90', 'display_name': 'hair'},
              91:  {'id': '91', 'display_name': 'toothbrush'},
              92:  {'id': '92', 'display_name': 'hair'},
              93:  {'id': '93', 'display_name': 'banner'},
              94:  {'id': '94', 'display_name': 'blanket'},
              95:  {'id': '95', 'display_name': 'branch'},
              96:  {'id': '96', 'display_name': 'bridge'},
              97:  {'id': '97', 'display_name': 'building'},
              98:  {'id': '98', 'display_name': 'bush'},
              99:  {'id': '99', 'display_name': 'cabinet'},
              100: {'id': '100', 'display_name': 'cage'},
              101: {'id': '101', 'display_name': 'cardboard'},
              102: {'id': '102', 'display_name': 'carpet'},
              103: {'id': '103', 'display_name': 'ceiling'},
              104: {'id': '104', 'display_name': 'ceiling'},
              105: {'id': '105', 'display_name': 'cloth'},
              106: {'id': '106', 'display_name': 'clothes'},
              107: {'id': '107', 'display_name': 'clouds'},
              108: {'id': '108', 'display_name': 'counter'},
              109: {'id': '109', 'display_name': 'cupboard'},
              110: {'id': '110', 'display_name': 'curtain'},
              111: {'id': '111', 'display_name': 'desk'},
              112: {'id': '112', 'display_name': 'dirt'},
              113: {'id': '113', 'display_name': 'door'},
              114: {'id': '114', 'display_name': 'fence'},
              115: {'id': '115', 'display_name': 'floor'},
              116: {'id': '116', 'display_name': 'floor'},
              117: {'id': '117', 'display_name': 'floor'},
              118: {'id': '118', 'display_name': 'floor'},
              119: {'id': '119', 'display_name': 'floor'},
              120: {'id': '120', 'display_name': 'flower'},
              121: {'id': '121', 'display_name': 'fog'},
              122: {'id': '122', 'display_name': 'food'},
              123: {'id': '123', 'display_name': 'fruit'},
              124: {'id': '124', 'display_name': 'furniture'},
              125: {'id': '125', 'display_name': 'grass'},
              126: {'id': '126', 'display_name': 'gravel'},
              127: {'id': '127', 'display_name': 'ground'},
              128: {'id': '128', 'display_name': 'hill'},
              129: {'id': '129', 'display_name': 'house'},
              130: {'id': '130', 'display_name': 'leaves'},
              131: {'id': '131', 'display_name': 'light'},
              132: {'id': '132', 'display_name': 'mat'},
              133: {'id': '133', 'display_name': 'metal'},
              134: {'id': '134', 'display_name': 'mirror'},
              135: {'id': '135', 'display_name': 'moss'},
              136: {'id': '136', 'display_name': 'mountain'},
              137: {'id': '137', 'display_name': 'mud'},
              138: {'id': '138', 'display_name': 'napkin'},
              139: {'id': '139', 'display_name': 'net'},
              140: {'id': '140', 'display_name': 'paper'},
              141: {'id': '141', 'display_name': 'pavement'},
              142: {'id': '142', 'display_name': 'pillow'},
              143: {'id': '143', 'display_name': 'plant'},
              144: {'id': '144', 'display_name': 'plastic'},
              145: {'id': '145', 'display_name': 'platform'},
              146: {'id': '146', 'display_name': 'playingfield'},
              147: {'id': '147', 'display_name': 'railing'},
              148: {'id': '148', 'display_name': 'railroad'},
              149: {'id': '149', 'display_name': 'river'},
              150: {'id': '150', 'display_name': 'road'},
              151: {'id': '151', 'display_name': 'rock'},
              152: {'id': '152', 'display_name': 'roof'},
              153: {'id': '153', 'display_name': 'rug'},
              154: {'id': '154', 'display_name': 'salad'},
              155: {'id': '155', 'display_name': 'sand'},
              156: {'id': '156', 'display_name': 'sea'},
              157: {'id': '157', 'display_name': 'shelf'},
              158: {'id': '158', 'display_name': 'sky'},
              159: {'id': '159', 'display_name': 'skyscraper'},
              160: {'id': '160', 'display_name': 'snow'},
              161: {'id': '161', 'display_name': 'solid'},
              162: {'id': '162', 'display_name': 'stairs'},
              163: {'id': '163', 'display_name': 'stone'},
              164: {'id': '164', 'display_name': 'straw'},
              165: {'id': '165', 'display_name': 'structural'},
              166: {'id': '166', 'display_name': 'table'},
              167: {'id': '167', 'display_name': 'tent'},
              168: {'id': '168', 'display_name': 'textile'},
              169: {'id': '169', 'display_name': 'towel'},
              170: {'id': '170', 'display_name': 'tree'},
              171: {'id': '171', 'display_name': 'vegetable'},
              172: {'id': '172', 'display_name': 'wall'},
              173: {'id': '173', 'display_name': 'wall'},
              174: {'id': '174', 'display_name': 'wall'},
              175: {'id': '175', 'display_name': 'wall'},
              176: {'id': '176', 'display_name': 'wall'},
              177: {'id': '177', 'display_name': 'wall'},
              178: {'id': '178', 'display_name': 'wall'},
              179: {'id': '179', 'display_name': 'water'},
              180: {'id': '180', 'display_name': 'waterdrops'},
              181: {'id': '181', 'display_name': 'window'},
              182: {'id': '182', 'display_name': 'window'},
              183: {'id': '183', 'display_name': 'wood'}}

    return labels