# 'Tensorflow Binary Fix'
# 
# This is a workaround for any issues 
# regarding the TF binary being built 
# for using SSE3, AVX, etc.
#
# Aditionally, you could also use
# 'export TF_CPP_MIN_LOG_LEVEL = 2'
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Numpy
import numpy as np

# Tensorflow
from tensorflow import convert_to_tensor

# Image processing
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

def parse_detections(image, confidence_threshold, detections, labels, args):

    """
    Parse the detections dictionary for each detection and evaluate
    each detection 

    Args:
        confidence_threshold: Threshold to filter out uncertain detections
        detections: Dictionary of detection results
        labels: Dictionary of the parsed labelmap file
        args: Dictionary of arguments
            - NET_FORMAT_NUMDET: Dictionary index to number of detections
            - NET_FORMAT_CLASSES: Dictionary index to class labels of detections
            - NET_FORMAT_SCORES: Dictionary index to detection confidence scores
            - NET_FORMAT_BOXES: Dictionary index to detection ROI boxes
            - LMF_FORMAT_CLASS: Dictionary index to class label string

    Returns:
        None

    Raises:
        N/A
    """

    # Iterate through each possible detection
    for idx in range(0, detections[args['net_format_numdet']]):

        # Get the class label index, the string label at 
        # that index, and the classification confidence 
        # value for the current detected object
        classification = detections[args['net_format_classes']][idx]
        confidence     = detections[args['net_format_scores']][idx]
        class_label    = labels[str(classification)][args['lmf_format_class']]

        # Check to make sure the detection confidence value
        # meets or exceedes the given threshold
        if confidence < confidence_threshold:
            continue

        # Calculate the detected object's 
        # normalized coordinates
        x_min = round(detections[args['net_format_boxes']][idx][1]*image.width)
        y_min = round(detections[args['net_format_boxes']][idx][0]*image.height)
        x_max = round(detections[args['net_format_boxes']][idx][3]*image.width)
        y_max = round(detections[args['net_format_boxes']][idx][2]*image.height)

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

def forward_pass(model, image, args):
    """
    Run a forward pass of an input image through
    the given model and return back the parsed 
    results

    Args:
        model: A model (saved_model format) to run data through
        image: Image input data to pass through a model
        args: Dictionary of arguments
            - NET_FORMAT_NUMDET: Dictionary index to number of detections
            - NET_FORMAT_CLASSES: Dictionary index to class labels of detections

    Returns:
        Parsed tensor of maps (dependant on model used)

    Raises:
        N/A
    """

    # Load in the image and then convert it
    image = np.array(image, dtype = 'uint8')
    input_tensor = convert_to_tensor(image)
    input_tensor = input_tensor[None, ...]

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
    num_detections = int(output.pop(args['net_format_numdet']))
    output = {key:val[0, : num_detections].numpy() for key, val in output.items()} 
    output[args['net_format_numdet']] = num_detections
    output[args['net_format_classes']] = output[args['net_format_classes']].astype(np.int64)
    return output