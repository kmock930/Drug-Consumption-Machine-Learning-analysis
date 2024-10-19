import random;
import scipy;
from scipy.stats import uniform, norm;

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
mlp_maxItr=500; # increase to avoid convergence warning
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
};
randomSearch_distributions={
    "C": uniform(loc=0, scale=4),
    "penalty": ['l2', 'l1'],
    "rvs": norm().rvs
};
random_state=42;