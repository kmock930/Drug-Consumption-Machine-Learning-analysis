import os;
import shutil;
import constants;

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

    if not os.path.exists(trained_dir):
        os.makedirs(trained_dir);

    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir);

    # List of all files in the root directory
    files = os.listdir(root_dir);

    # Loop through the files and move them to the appropriate folder
    for file in files:
        if file.startswith(f"{dataset_name}_model") and file.endswith('.pkl'):
            if 'pretrained' in file:
                # Move to pretrained folder
                src_path = os.path.join(root_dir, file);
                dest_path = os.path.join(pretrained_dir, file);
                shutil.move(src_path, dest_path);
                print(f"Moved {file} to {pretrained_dir}");
            else:
                # Move to trained folder
                src_path = os.path.join(root_dir, file);
                dest_path = os.path.join(trained_dir, file);
                shutil.move(src_path, dest_path);
                print(f"Moved {file} to {trained_dir}");

print("Organization complete!");