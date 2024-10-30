'''
This is a script that executes to 
move specific .pkl model files into proper directories.
'''
import os;
import shutil;
import pickle;
import constants;

import os;
import joblib;
import traceback;

def unpack(filepath):
    with open(filepath, 'rb') as file:
        return joblib.load(file);

def organize():
    # Define the root directory where the model files are located
    root_dir = './';

    # Dataset name extracted from file names (in this case "choc")
    datasets = [constants.choco_dataset, constants.mushrooms_dataset];

    for dataset_name in datasets:
        # Create dataset directory
        dataset_dir = os.path.join(root_dir, dataset_name);
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir);

        # Create subdirectories for trained and pre-trained models
        trained_dir = os.path.join(dataset_dir, 'posttrained');
        pretrained_dir = os.path.join(dataset_dir, 'pretrained');
        train_set_dir = os.path.join(dataset_dir, 'Training Set');
        test_set_dir = os.path.join(dataset_dir, 'Test Set');

        if not os.path.exists(trained_dir):
            os.makedirs(trained_dir);

        if not os.path.exists(pretrained_dir):
            os.makedirs(pretrained_dir);
        
        if not os.path.exists(train_set_dir):
            os.makedirs(train_set_dir);
        
        if not os.path.exists(test_set_dir):
            os.makedirs(test_set_dir);

        # List of all files in the root directory
        files = os.listdir(root_dir);

        # Loop through the files and move them to the appropriate folder
        for file in files:
            if (file.endswith('.pkl') and dataset_name in file): # organize all .pkl files
                if (file.startswith(f"{dataset_name}_model")):
                    if 'pretrained' in file:
                        # Move to pretrained folder
                        src_path = os.path.join(root_dir, file);
                        dest_path = os.path.join(pretrained_dir, file);
                        shutil.move(src_path, dest_path);
                        print(f"Moved {file} to {os.path.join(pretrained_dir)}");
                    elif 'posttrained' in file:
                        # Move to trained folder
                        src_path = os.path.join(root_dir, file);
                        dest_path = os.path.join(trained_dir, file);
                        shutil.move(src_path, dest_path);
                        print(f"Moved {file} to {os.path.join(trained_dir)}");
                elif ('train-set' in file): # training set data
                    src_path = os.path.join(root_dir, file);
                    dest_path = os.path.join(train_set_dir, file);
                    shutil.move(src_path, dest_path);
                    print(f"Moved {file} to {os.path.join(train_set_dir)}");
                elif ('test-set' in file): # test set data
                    src_path = os.path.join(root_dir, file);
                    dest_path = os.path.join(test_set_dir, file);
                    shutil.move(src_path, dest_path);
                    print(f"Moved {file} to {os.path.join(test_set_dir)}");


    print("Organization complete!");