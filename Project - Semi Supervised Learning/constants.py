# directories
ASM1_DIR = "Assignment 1 - Model Comparison";
ASM2_DIR = "Assignment 2 - Explainable AI";
# datasets
MUSH_DIR = "mushrooms";
# data / model type
POSTTRAINED_DIR = "posttrained";
TRAIN_DIR = "Training Set";
TEST_DIR = "Test Set";
COLUMNS_DIR = "Columns";

# model re-used
MODEL_FILENAME = "mushrooms_model_Gradient_Boosting_posttrained.pkl";
MODEL_SELFTRAIN_FILENAME = "mushrooms_model_Gradient_Boosting_selftrained.pkl";
MODEL_COTRAIN_FILENAME = "mushrooms_model_Gradient_Boosting_cotrained.pkl";
MODEL_COTRAIN_1_FILENAME = "mushrooms_model_Gradient_Boosting_cotrained_1.pkl";
MODEL_COTRAIN_2_FILENAME = "mushrooms_model_Gradient_Boosting_cotrained_2.pkl";
MODEL_SEMIBOOST_FILENAME = "mushrooms_model_SemiBoost.pkl";
MODEL_DIR = "model";

# data re-used
X_TRAIN_FILENAME = "mushrooms_train-set_samples.pkl";
Y_TRAIN_FILENAME = "mushrooms_train-set_labels.pkl";
X_TEST_FILENAME = "mushrooms_test-set_samples.pkl";
Y_TEST_FILENAME = "mushrooms_test-set_labels.pkl";
# normalized data re-used
X_TRAIN_NORMALIZED_FILENAME = "mushrooms_train-set_samples_normalized.pkl";
Y_TRAIN_NORMALIZED_FILENAME = "mushrooms_train-set_labels_normalized.pkl";
X_TEST_NORMALIZED_FILENAME = "mushrooms_test-set_samples_normalized.pkl";
Y_TEST_NORMALIZED_FILENAME = "mushrooms_test-set_labels_normalized.pkl";
# labelled data
X_TRAIN_LABELLED_FILENAME = "mushrooms_train-set_samples_labelled.pkl";
Y_TRAIN_LABELLED_FILENAME = "mushrooms_train-set_labels_labelled.pkl";
# unlabeled data
X_TRAIN_UNLABELLED_FILENAME = "mushrooms_train-set_samples_unlabelled.pkl";
Y_TRAIN_UNLABELLED_FILENAME = "mushrooms_train-set_labels_unlabelled.pkl";
# Pseudo-labelled data
Y_TRAIN_PSEUDO_FILENAME = "mushrooms_train-set_samples_pseudo.pkl";
# co-training data
CO_TRAINING_BEST_FEATURE_SPLIT_RESULTS_FILENAME = "mushrooms_cotraining_best_feature_split_results.pkl";
# columns file
COLUMNS_FILENAME = "mushrooms_columns.pkl";

# Labelled-Unlabelled portion
LABELLED_PORTION = 0.2;

# co training feature split
CO_TRAINING_FEATURE_SPLIT = 0.5;

# Semi Boost parameters
N_NEIGHBORS = 5;
N_ESTIMATORS = 100;
LEARNING_RATE = 0.1;

RANDOM_STATE = 42;