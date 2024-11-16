import random;
from scipy.stats import randint, uniform;

tree_entropyCriterion = "entropy";
splitter="random";
descisionTree="decision tree";
randForest="random forest";
svm="SVM";
svc_kernel='rbf'; # the data is obviously non-linear, hence we use SVC with RBF kernel
svc_gamma='scale'; # uses 1 / (n_features * X.var())
gradientBoost="gradient boosting";
gradient_Loss="log_loss";
gradient_LearningRate=0.1;
gradient_estimators=100;
gradient_subsamples=1.0;
gradient_criterion="friedman_mse"; # Fredmen's test will be used (with mean squared error)
mlp="multi-layer perceptron (MLP)";
mlp_hidden_layer_size=1885 * 2 - 1; # maxed the size: number of samples x 2 - 1
mlp_activation="relu"; # Rectified Linear Unit: no bound for input
mlp_solver="adam"; # stochastic gradient-based optimizer
mlp_alpha=0.0001;
mlp_LearningRate="adaptive"; # keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing
mlp_maxItr=2000; # increase to avoid convergence warning
knn="k‐nearest neighbour (k-NN) classifier";
knn_neighbors=random.randrange(3,9,2); # randomize an odd number for k
knn_weights="distance"; # weight points by the inverse of their distance
knn_algorithm="auto"; # let the library decide for us based on our fitting data
knn_distMetric=2; # we use Euclidian distance here
knn_metric="minkowski";
knn_jobs=-1; # using all processors
column_names_orig = [
    "ID",
    "age",
    "gender",
    "education",
    "country",
    "ethnicity",
    "nscore",
    "escore",
    "oscore",
    "ascore",
    "cscore",
    "impuslive",
    "ss",
    "alcohol",
    "amphet",
    "amyl",
    "benzos",
    "caff",
    "cannabis",
    "choc",
    "coke",
    "crack",
    "ecstasy",
    "heroin",
    "ketamine",
    "legalh",
    "lsd",
    "meth",
    "mushrooms",
    "nicotine",
    "semer",
    "vsa" 
];
choco_dataset="choc";
choco_dataset_fullname="chocolate";
mushrooms_dataset="mushrooms";
# filepaths
filepaths = {
    # choc posttrained
    "choc_posttrained_decisionTree": ".\choc\posttrained\choc_model_decisionTree_posttrained.pkl",
    "choc_posttrained_gradientBoost": ".\choc\posttrained\choc_model_Gradient_Boosting_posttrained.pkl",
    "choc_posttrained_KNN": ".\choc\posttrained\choc_model_KNN_posttrained.pkl",
    "choc_posttrained_MLP": ".\choc\posttrained\choc_model_MLP_posttrained.pkl",
    "choc_posttrained_randomForest": ".\choc\posttrained\choc_model_randomForest_posttrained.pkl",
    "choc_posttrained_SVC": ".\choc\posttrained\choc_model_SVC_RBF_posttrained.pkl",
    # choc pretrained
    "choc_pretrained_decisionTree": ".\choc\pretrained\choc_model_decisionTree_pretrained.pkl",
    "choc_pretrained_gradientBoost": ".\choc\pretrained\choc_model_Gradient_Boosting_pretrained.pkl",
    "choc_pretrained_KNN": ".\choc\pretrained\choc_model_KNN_pretrained.pkl",
    "choc_pretrained_MLP": ".\choc\pretrained\choc_model_MLP_pretrained.pkl",
    "choc_pretrained_randomForest": ".\choc\pretrained\choc_model_randomForest_pretrained.pkl",
    "choc_pretrained_SVC": ".\choc\pretrained\choc_model_SVC_RBF_pretrained.pkl",
    #choc test set
    "choc_test-set_labels": ".\choc\Test Set\choc_test-set_labels.pkl",
    "choc_test-set_samples": ".\choc\Test Set\choc_test-set_samples.pkl",
    # choc training set
    "choc_train-set_labels": ".\choc\Training Set\choc_train-set_labels.pkl",
    'choc_train-set_samples': ".\choc\Training Set\choc_train-set_samples.pkl",
    # mushrooms posttrained
    "mushrooms_posttrained_decisionTree": ".\mushrooms\posttrained\mushrooms_model_decisionTree_posttrained.pkl",
    "mushrooms_posttrained_gradientBoost": ".\mushrooms\posttrained\mushrooms_model_Gradient_Boosting_posttrained.pkl",
    "mushrooms_posttrained_KNN": ".\mushrooms\posttrained\mushrooms_model_KNN_posttrained.pkl",
    "mushrooms_posttrained_MLP": ".\mushrooms\posttrained\mushrooms_model_MLP_posttrained.pkl",
    "mushrooms_posttrained_randomForest": ".\mushrooms\posttrained\mushrooms_model_randomForest_posttrained.pkl",
    "mushrooms_posttrained_SVC": ".\mushrooms\posttrained\mushrooms_model_SVC_RBF_posttrained.pkl",
    # mushrooms pretrained
    "mushrooms_pretrained_decisionTree": ".\mushrooms\pretrained\mushrooms_model_decisionTree_pretrained.pkl",
    "mushrooms_pretrained_gradientBoost": ".\mushrooms\pretrained\mushrooms_model_Gradient_Boosting_pretrained.pkl",
    "mushrooms_pretrained_KNN": ".\mushrooms\pretrained\mushrooms_model_KNN_pretrained.pkl",
    "mushrooms_pretrained_MLP": ".\mushrooms\pretrained\mushrooms_model_MLP_pretrained.pkl",
    "mushrooms_pretrained_randomForest": ".\mushrooms\pretrained\mushrooms_model_randomForest_pretrained.pkl",
    "mushrooms_pretrained_SVC": ".\mushrooms\pretrained\mushrooms_model_SVC_RBF_pretrained.pkl",
    #mushrooms test set
    "mushrooms_test-set_labels": ".\mushrooms\Test Set\mushrooms_test-set_labels.pkl",
    "mushrooms_test-set_samples": ".\mushrooms\Test Set\mushrooms_test-set_samples.pkl",
    # mushrooms training set
    "mushrooms_train-set_labels": ".\mushrooms\Training Set\mushrooms_train-set_labels.pkl",
    'mushrooms_train-set_samples': ".\mushrooms\Training Set\mushrooms_train-set_samples.pkl",
    # choc undersampled
    "choc_posttrained_undersampled_decisionTree": ".\choc\posttrained\choc_model_decisionTree_posttrained_undersampled.pkl",
    "choc_posttrained_undersampled_randomForest": ".\choc\posttrained\choc_model_randomForest_posttrained_undersampled.pkl",
    "choc_posttrained_undersampled_SVC": ".\choc\posttrained\choc_model_SVC_RBF_posttrained_undersampled.pkl",
    "choc_posttrained_undersampled_gradientBoost": ".\choc\posttrained\choc_model_Gradient_Boosting_posttrained_undersampled.pkl",
    "choc_posttrained_undersampled_MLP": ".\choc\posttrained\choc_model_MLP_posttrained_undersampled.pkl",
    "choc_posttrained_undersampled_KNN": ".\choc\posttrained\choc_model_KNN_posttrained_undersampled.pkl",
    # choc oversampled
    "choc_posttrained_oversampled_decisionTree": ".\choc\posttrained\choc_model_decisionTree_posttrained_oversampled.pkl",
    "choc_posttrained_oversampled_randomForest": ".\choc\posttrained\choc_model_randomForest_posttrained_oversampled.pkl",
    "choc_posttrained_oversampled_SVC": ".\choc\posttrained\choc_model_SVC_RBF_posttrained_oversampled.pkl",
    "choc_posttrained_oversampled_gradientBoost": ".\choc\posttrained\choc_model_Gradient_Boosting_posttrained_oversampled.pkl",
    "choc_posttrained_oversampled_MLP": ".\choc\posttrained\choc_model_MLP_posttrained_oversampled.pkl",
    "choc_posttrained_oversampled_KNN": ".\choc\posttrained\choc_model_KNN_posttrained_oversampled.pkl",
    # choc combinedsampled
    "choc_posttrained_combinedsampled_decisionTree": ".\choc\posttrained\choc_model_decisionTree_posttrained_combinedsampled.pkl",
    "choc_posttrained_combinedsampled_randomForest": ".\choc\posttrained\choc_model_randomForest_posttrained_combinedsampled.pkl",
    "choc_posttrained_combinedsampled_SVC": ".\choc\posttrained\choc_model_SVC_RBF_posttrained_combinedsampled.pkl",
    "choc_posttrained_combinedsampled_gradientBoost": ".\choc\posttrained\choc_model_Gradient_Boosting_posttrained_combinedsampled.pkl",
    "choc_posttrained_combinedsampled_MLP": ".\choc\posttrained\choc_model_MLP_posttrained_combinedsampled.pkl",
    "choc_posttrained_combinedsampled_KNN": ".\choc\posttrained\choc_model_KNN_posttrained_combinedsampled.pkl",
    # mushrooms undersampled
    "mushrooms_posttrained_undersampled_decisionTree": ".\mushrooms\posttrained\mushrooms_model_decisionTree_posttrained_undersampled.pkl",
    "mushrooms_posttrained_undersampled_randomForest": ".\mushrooms\posttrained\mushrooms_model_randomForest_posttrained_undersampled.pkl",
    "mushrooms_posttrained_undersampled_SVC": ".\mushrooms\posttrained\mushrooms_model_SVC_RBF_posttrained_undersampled.pkl",
    "mushrooms_posttrained_undersampled_gradientBoost": ".\mushrooms\posttrained\mushrooms_model_Gradient_Boosting_posttrained_undersampled.pkl",
    "mushrooms_posttrained_undersampled_MLP": ".\mushrooms\posttrained\mushrooms_model_MLP_posttrained_undersampled.pkl",
    "mushrooms_posttrained_undersampled_KNN": ".\mushrooms\posttrained\mushrooms_model_KNN_posttrained_undersampled.pkl",
    # mushrooms oversampled
    "mushrooms_posttrained_oversampled_decisionTree": ".\mushrooms\posttrained\mushrooms_model_decisionTree_posttrained_oversampled.pkl",
    "mushrooms_posttrained_oversampled_randomForest": ".\mushrooms\posttrained\mushrooms_model_randomForest_posttrained_oversampled.pkl",
    "mushrooms_posttrained_oversampled_SVC": ".\mushrooms\posttrained\mushrooms_model_SVC_RBF_posttrained_oversampled.pkl",
    "mushrooms_posttrained_oversampled_gradientBoost": ".\mushrooms\posttrained\mushrooms_model_Gradient_Boosting_posttrained_oversampled.pkl",
    "mushrooms_posttrained_oversampled_MLP": ".\mushrooms\posttrained\mushrooms_model_MLP_posttrained_oversampled.pkl",
    "mushrooms_posttrained_oversampled_KNN": ".\mushrooms\posttrained\mushrooms_model_KNN_posttrained_oversampled.pkl",
    # mushrooms combinedsampled
    "mushrooms_posttrained_combinedsampled_decisionTree": ".\mushrooms\posttrained\mushrooms_model_decisionTree_posttrained_combinedsampled.pkl",
    "mushrooms_posttrained_combinedsampled_randomForest": ".\mushrooms\posttrained\mushrooms_model_randomForest_posttrained_combinedsampled.pkl",
    "mushrooms_posttrained_combinedsampled_SVC": ".\mushrooms\posttrained\mushrooms_model_SVC_RBF_posttrained_combinedsampled.pkl",
    "mushrooms_posttrained_combinedsampled_gradientBoost": ".\mushrooms\posttrained\mushrooms_model_Gradient_Boosting_posttrained_combinedsampled.pkl",
    "mushrooms_posttrained_combinedsampled_MLP": ".\mushrooms\posttrained\mushrooms_model_MLP_posttrained_combinedsampled.pkl",
    "mushrooms_posttrained_combinedsampled_KNN": ".\mushrooms\posttrained\mushrooms_model_KNN_posttrained_combinedsampled.pkl",
};
randomSearch_distributions={
    'max_depth': randint(1, 20).rvs(size=100),
    'min_samples_split': uniform(0.1, 0.9)
};
randomSearch_distributions_MLP={
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [2000, 2000]
};
randomSearch_distributions_distributions_SVC={
    'C': uniform(0.1, 10),  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel types
    'gamma': ['scale', 'auto'] + list(uniform(0.01, 0.1).rvs(size=5)),  # Gamma for rbf/poly kernels
    'degree': randint(2, 5)  # Only for polynomial kernels
};
randomSearch_distributions_distributions_KNN={
    'n_neighbors': randint(1, 30, 2),  # Number of neighbors to use (odd numbers only)
    'weights': ['uniform', 'distance'],  # Weight function
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
    'leaf_size': randint(1, 50),  # Leaf size for BallTree or KDTree
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metric
}
random_state=42;
underSamplePercent=0.5;
oversampledPercent=1.0;
undersampled="undersampled";
oversampled="oversampled";
combinedsampled="combinedsampled";