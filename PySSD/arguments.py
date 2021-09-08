import argparse

def get_parser():
    
    # Define argument parser
    parser = argparse.ArgumentParser(prog = "PySSD Driver", description = "Python SSD Driver Program")

    # Parse each path given as well as the 
    # classification confidence threshold
    parser.add_argument(
        "--img_path", 
        default = "Images/DogPark3.jpg", 
        help = "File Path: Input image"
    )
    parser.add_argument(
        "--net_path", 
        default = "./", 
        help = "File Path: Saved network model"
    )
    parser.add_argument(
        "--lmf_path", 
        default = "mscoco_label_map.pbtxt", 
        help = "File Path: Label map"
    )
    parser.add_argument(
        "--conf_threshold", 
        default = 0.60, 
        help = "Classification threshold"
    )

    # Parse the labelmap formatting given
    parser.add_argument(
        "--lmf_format_id",
        default = "id",
        help = "Label Map Format: Unique class identifier"
    )
    parser.add_argument(
        "--lmf_format_class",
        default = "display_name",
        help = "Label Map Format: Class name"
    )
    parser.add_argument(
        "--lmf_format_misc",
        default = "name",
        help = "Label Map Format: Misc. information"
    )

    # Parse the network output formatting given
    parser.add_argument(
        "--net_format_numdet",
        default = "num_detections",
        help = "Network Output Format: Number of detections"
    )
    parser.add_argument(
        "--net_format_boxes",
        default = "detection_boxes",
        help = "Network Output Format: Detection ROI boxes"
    )
    parser.add_argument(
        "--net_format_classes",
        default = "detection_classes",
        help = "Network Output Format: Detected classes"
    )
    parser.add_argument(
        "--net_format_scores",
        default = "detection_scores",
        help = "Network Output Format: Detection confidence values"
    )

    return parser