import random;
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
mlp_hidden_layer_size=1885 * 2 - 1; # < input size + output size
mlp_activation="relu"; # no bound for input
mlp_solver="adam"; # stochastic gradient-based optimizer
mlp_alpha=0.0001;
mlp_LearningRate="adaptive"; # keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing
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