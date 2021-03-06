# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os
from cntk import load_model
from TransferLearning import *


# define base model location and characteristics
base_folder = os.path.dirname(os.path.abspath(__file__))
base_model_file = os.path.join(base_folder, "..", "PretrainedModels", "ResNet_18.model")
new_model_file = os.path.join(base_folder, "Output", "TransferLearning.model")
feature_node_name = "features"
last_hidden_node_name = "z.x"
image_height = 224
image_width = 224
num_channels = 3

# define data location and characteristics
train_image_folder = os.path.join(base_folder, "..", "DataSets", "Animals", "Train")
test_image_folder = os.path.join(base_folder, "..", "DataSets", "Animals", "Test")
file_endings = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']


def create_map_file_from_folder(root_folder, class_mapping, include_unknown=False):
    map_file_name = os.path.join(root_folder, "map.txt")
    with open(map_file_name , 'w') as map_file:
        for class_id in range(0, len(class_mapping)):
            folder = os.path.join(root_folder, class_mapping[class_id])
            if os.path.exists(folder):
                for entry in os.listdir(folder):
                    filename = os.path.join(folder, entry)
                    if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                        map_file.write("{0}\t{1}\n".format(filename, class_id))

        if include_unknown:
            for entry in os.listdir(root_folder):
                filename = os.path.join(root_folder, entry)
                if os.path.isfile(filename) and os.path.splitext(filename)[1] in file_endings:
                    map_file.write("{0}\t-1\n".format(filename))

    return map_file_name


def create_class_mapping_from_folder(root_folder):
    classes = []
    for _, directories, _ in os.walk(root_folder):
        for directory in directories:
            classes.append(directory)
    return np.asarray(classes)


def format_output_line(img_name, true_class, probs, class_mapping, top_n=3):
    class_probs = np.column_stack((probs, class_mapping)).tolist()
    class_probs.sort(key=lambda x: float(x[0]), reverse=True)
    top_n = min(top_n, len(class_mapping)) if top_n > 0 else len(class_mapping)
    true_class_name = class_mapping[true_class] if true_class >= 0 else 'unknown'
    line = '[{"class": "%s", "predictions": {' % true_class_name
    for i in range(0, top_n):
        line = '%s"%s":%.3f, ' % (line, class_probs[i][1], float(class_probs[i][0]))
    line = '%s}, "image": "%s"}]\n' % (line[:-2], img_name.replace('\\', '/'))
    return line


if __name__ == '__main__':
    # check for model and data existence
    if not (os.path.exists(base_model_file) and os.path.exists(train_image_folder) and os.path.exists(test_image_folder)):
        print("Please run 'python install_data_and_model.py' first to get the required data and model.")
        exit(0)

    # get class mapping and map files from train and test image folder
    class_mapping = create_class_mapping_from_folder(train_image_folder)
    train_map_file = create_map_file_from_folder(train_image_folder, class_mapping)
    test_map_file = create_map_file_from_folder(test_image_folder, class_mapping, include_unknown=True)

    # train
    trained_model = train_model(base_model_file, feature_node_name, last_hidden_node_name,
                                image_width, image_height, num_channels,
                                len(class_mapping), train_map_file, num_epochs=30, freeze=True)
    trained_model.save(new_model_file)
    print("Stored trained model at %s" % tl_model_file)

    # evaluate test images
    # trained_model = load_model(new_model_file)
    results_file = os.path.join(base_folder, "Output", "predictions.txt")
    with open(results_file, 'w') as output_file:
        with open(test_map_file, "r") as input_file:
            for line in input_file:
                tokens = line.rstrip().split('\t')
                img_file = tokens[0]
                true_label = int(tokens[1])
                probs = eval_single_image(trained_model, img_file, image_width, image_height)

                formatted_line = format_output_line(img_file, true_label, probs, class_mapping)
                output_file.write(formatted_line)

    print("Done. Wrote output to %s" % results_file)
