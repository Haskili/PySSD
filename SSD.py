from PySSD.arguments import get_parser
from PySSD.labelmap import (get_LM_PBTXT, get_LM_STATIC)
from PySSD.detection import (forward_pass, parse_detections)
from tensorflow import saved_model
from PIL import Image

# NOTE: This was made with the models like the (MobileNetV2) SSD 
#       hosted on the Tensorflow Hub website in mind;
#       (e.g. https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)
#
#       Certain things such as how the model output is accessed would
#       change depending on what model you're using, please be careful
#       to check the documented model output before using this program

if __name__ == "__main__":

    # Parse the arguments from the command line
    args = vars(get_parser().parse_args())

    # Get the label map either by loading in or 
    # generating one from a static dictionary
    print(f"Generating label map from {args['lmf_path']}")
    labels = {}
    if (args['lmf_path'][-5:] == "pbtxt"):
        labels = get_LM_PBTXT(args['lmf_path'], args['lmf_format_id'])

    else:
        labels = get_LM_STATIC()

    # Load the SSD model
    print(f"Loading model at {args['net_path']}")
    model = saved_model.load(args['net_path'], [saved_model.SERVING])

    # Load the image and run it through the model,
    # checking for all recognizable objects in the image
    print(f"Running input {args['img_path']} through the model\n")
    input_IMG  = Image.open(args['img_path'])
    detections = forward_pass(model, input_IMG, args)

    # Iterate through the results and parse the data
    # from those results, highlighting each detected
    # object's Region-of-Interest (ROI) and labeling 
    # each detected object with it's appropriate class
    parse_detections(input_IMG, args['conf_threshold'], detections, labels, args)

    # Save the processed image
    input_IMG.save("./result.png", format = 'PNG')
    print(f"Finished with input {args['img_path']}")