from PySSD import *

# NOTE: This was made with the models like the (MobileNetV2) SSD 
#       hosted on the Tensorflow Hub website in mind;
#       (e.g. https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2)
#
#       Certain things such as how the model output is accessed would
#       change depending on what model you're using, please be careful
#       to check the documented model output before using this program

if __name__ == "__main__":

    # Get the label map specified at 'LMF_PATH'
    print(f'Generating label map from "{LMF_PATH}"')
    labels = {}
    if (LMF_PATH[-5:] == "pbtxt"):
        labels = get_LM_PBTXT(LMF_PATH, LMF_FORMAT_ID)

    else:
        labels = get_LM_STATIC()

    # Load the SSD model at 'NET_PATH'
    print(f'Loading model at "{NET_PATH}"')
    model = tf.saved_model.load(NET_PATH, [tf.saved_model.SERVING])

    # Load the image and run it through the model,
    # checking for all recognizable objects in the image
    print(f'Running input "{IMG_PATH}" through the model\n')
    input_IMG  = Image.open(IMG_PATH)
    detections = forward_pass(model, input_IMG)

    # Iterate through the results and parse the data
    # from those results, highlighting each detected
    # object's Region-of-Interest (ROI) and labeling 
    # each detected object with it's appropriate class
    parse_detections(input_IMG, CONF_THRESHOLD, 
                            detections, labels)


    # Save the processed image
    input_IMG.save("./result.png", format = 'PNG')
    print(f'Finished with input "{IMG_PATH}"')