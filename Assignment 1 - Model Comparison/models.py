import numpy as np;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier;
from sklearn.svm import SVC;
from sklearn.neural_network import MLPClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import LabelEncoder;
from sklearn.model_selection import RandomizedSearchCV;
from imblearn.pipeline import Pipeline;
import joblib;
import constants;
import traceback;
import sys;

'''
@summary A class which encapsulates various models which have been initialized
'''
class Models:
    # classifiers
    decisionTree_clf:DecisionTreeClassifier;
    randomForest_clf:RandomForestClassifier;
    svm_clf:SVC;
    gradientBoost_clf:GradientBoostingClassifier;
    mlp_clf:MLPClassifier;
    knn_clf:KNeighborsClassifier;

    # data
    X_train: np.ndarray;
    y_train: np.ndarray;
    X_test: np.ndarray;
    y_test: np.ndarray;

    def __init__(self, decisionTree_clf:DecisionTreeClassifier=None, randomForest_clf:RandomForestClassifier=None, svm_clf:SVC=None, gradientBoost_clf:GradientBoostingClassifier=None, mlp_clf:MLPClassifier=None, knn_clf:KNeighborsClassifier=None):
        if (decisionTree_clf == None):
            self.decisionTree_clf = DecisionTreeClassifier(
                criterion=constants.tree_entropyCriterion,
                splitter=constants.splitter,
                max_depth=None
            );
        else:
            self.decisionTree_clf = decisionTree_clf;

        if (randomForest_clf == None):
            self.randomForest_clf = RandomForestClassifier(
                criterion=constants.tree_entropyCriterion,
                bootstrap=True # the dataset is split into different trees
            );
        else:
            self.randomForest_clf = randomForest_clf;

        if (svm_clf == None):
            self.svm_clf = SVC(
                C=1.0,
                kernel=constants.svc_kernel,
                degree=3, # Degree of the polynomial kernel function
                gamma=constants.svc_gamma
            );
        else:
            self.svm_clf = svm_clf;

        if (gradientBoost_clf == None):
            self.gradientBoost_clf = GradientBoostingClassifier(
                loss=constants.gradient_Loss, # Gradient Boost algorithm
                learning_rate=constants.gradient_LearningRate,
                n_estimators=constants.gradient_estimators, # number of boosting stages to perform
                subsample=constants.gradient_subsamples, # fraction of samples to be used for fitting the individual base learners
                criterion=constants.gradient_criterion
            );
        else:
            self.gradientBoost_clf = gradientBoost_clf;

        if (mlp_clf == None):
            self.mlp_clf = MLPClassifier(
                hidden_layer_sizes=constants.mlp_hidden_layer_size,
                activation=constants.mlp_activation, # Activation function
                solver=constants.mlp_solver, # weight optimization
                alpha=constants.mlp_alpha, # Strength of the L2 regularization
                learning_rate=constants.mlp_LearningRate,
                max_iter=constants.mlp_maxItr,
                shuffle=True # shuffle samples in each iteration
            );
        else:
            self.mlp_clf = mlp_clf;
    
        if (knn_clf == None):
            self.knn_clf = KNeighborsClassifier(
                n_neighbors=constants.knn_neighbors,
                weights=constants.knn_weights, # Weight function used in prediction
                algorithm=constants.knn_algorithm,
                p=constants.knn_distMetric,
                metric=constants.knn_metric,
                n_jobs=constants.knn_jobs
            );
        else:
            self.knn_clf = knn_clf;

    '''
    @param args: pass as many string as possible to specify which model to output
    Possible models:
    - 'decision tree'
    - 'random forest'
    - 'SVM'
    - 'gradient boosting'
    - 'multi-layer perceptron (MLP)'
    - 'k‐nearest neighbour (k-NN) classifier'
    '''
    def getModels(self, *args):
        res = {};
        if (args == {} or constants.descisionTree in args):
            res.update({constants.descisionTree: self.decisionTree_clf});
        if (args == {} or constants.randForest in args):
            res.update({constants.randForest: self.randomForest_clf});
        if (args == {} or constants.svm in args):
            res.update({constants.svm: self.svm_clf});
        if (args == {} or constants.gradientBoost in args):
            res.update({constants.gradientBoost: self.gradientBoost_clf});
        if (args == {} or constants.mlp in args):
            res.update({constants.mlp: self.mlp_clf});
        if (args == {} or constants.knn in args):
            res.update({constants.knn: self.knn_clf});
               
        return res;

    def set_X_train(self, X:np.ndarray):
        self.X_train = X;
    
    def set_y_train(self, y:np.ndarray):
        self.y_train = y;
    
    def set_X_test(self, X: np.ndarray):
        self.X_test = X;
    
    def set_y_test(self, y: np.ndarray):
        self.y_test = y;
    
    '''
    @summary A function that returns the prediction results in a boolean matrix.
    @param model - one of the 6 designated classifiers
    @param X_test: numpy array - a sample array for preddiction
    @param y_test: numpy array - an array of actual labels for evaluating the prediction
    @return numpy array - prediction results
    '''
    def predict(model: DecisionTreeClassifier | RandomForestClassifier | SVC | GradientBoostingClassifier | MLPClassifier | KNeighborsClassifier = None, X_test:np.ndarray = None, y_test: np.ndarray = None):
        # ensure all parameters are supplied
        if (model == None):
            raise ValueError("Please specify a model for prediction.");
        if (y_test is None):
            raise ValueError("Please specify the actual labels for prediction.");
        if (X_test is None):
            raise ValueError("Please specify the samples for testing.");
        # Prediction
        y_pred:np.ndarray = model.predict(X_test);
        return y_pred;
    
    def normalize(self):
        scaler = StandardScaler(); # uses the z-score to calibrate
        le = LabelEncoder(); # convert categorical labels into numeric representation
        self.X_train = scaler.fit_transform(self.X_train);
        self.X_test = scaler.transform(self.X_test);
        self.y_train = le.fit_transform(self.y_train);
        self.y_test = le.transform(self.y_test);
    
    def train(self, model: DecisionTreeClassifier | RandomForestClassifier | SVC | GradientBoostingClassifier | MLPClassifier | KNeighborsClassifier | Pipeline = None, dataset=constants.choco_dataset, isSampled:str=None):
        try:
            exc_info = sys.exc_info();

            if not model:
                print('Training all models');
                self.decisionTree_clf.fit(self.X_train, self.y_train);
                self.randomForest_clf.fit(self.X_train, self.y_train);
                self.svm_clf.fit(self.X_train, self.y_train);
                self.gradientBoost_clf.fit(self.X_train, self.y_train);
                self.mlp_clf.fit(self.X_train, self.y_train);
                self.knn_clf.fit(self.X_train, self.y_train);
            else:
                print('Training the specific model')
                model.fit(self.X_train, self.y_train);

            # save all trained models
            self.saveModels(isTrained=True, isSampled=isSampled, dataset=dataset);
        except:
            print("Failed to train a model.");
            traceback.print_exc();
            return False;
        print("All models are completely trained.");
        return True;

    def saveModels(self, isTrained=False, isSampled:str=None, dataset=constants.choco_dataset, **args):
        try:
            # format part of the filename
            # by deciding whether the saved model file is trained or untrained
            isTrained_string = "";
            if (isTrained == True):
                isTrained_string = '_posttrained';
            else:
                isTrained_string = '_pretrained';
            
            if (isSampled != None):
                isSampled = '_' + isSampled;
            else:
                isSampled = '';
            
            if (args== {} or constants.descisionTree in args):
                joblib.dump(self.decisionTree_clf, dataset + '_model_decisionTree' + isTrained_string + isSampled + '.pkl');
            if (args== {} or constants.randForest in args):
                joblib.dump(self.randomForest_clf, dataset + '_model_randomForest' + isTrained_string + isSampled + '.pkl');
            if (args== {} or constants.svm in args):
                joblib.dump(self.svm_clf, dataset + '_model_SVC_RBF' + isTrained_string + isSampled + '.pkl');
            if (args== {} or constants.gradientBoost in args):
                joblib.dump(self.gradientBoost_clf, dataset + '_model_Gradient_Boosting' + isTrained_string + isSampled + '.pkl');
            if (args== {} or constants.mlp in args):
                joblib.dump(self.mlp_clf, dataset + '_model_MLP' + isTrained_string + isSampled + '.pkl');
            if (args== {} or constants.knn in args):
                joblib.dump(self.knn_clf, dataset + '_model_KNN' + isTrained_string + isSampled + '.pkl');
        except:
            traceback.print_exc();
            return False;
        return True;

    # random search approach
    def paramTuning(model: DecisionTreeClassifier | RandomForestClassifier | SVC | GradientBoostingClassifier | MLPClassifier | KNeighborsClassifier = None):
        if (isinstance(model, SVC)):
            # params suitable for this kernel based model
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=constants.randomSearch_distributions_distributions_SVC,
                random_state=constants.random_state
            );
        elif (isinstance(model, KNeighborsClassifier)):
            # parameters for this non-parametric model (neither kernel based nor tree based)
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=constants.randomSearch_distributions_distributions_KNN,
                random_state=constants.random_state
            );
        else:
            return RandomizedSearchCV(
                estimator=model,
                param_distributions=constants.randomSearch_distributions,
                random_state=constants.random_state
            );