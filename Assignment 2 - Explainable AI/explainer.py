import shap;
import shap.maskers
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier;
from sklearn.svm import SVC;
from sklearn.neural_network import MLPClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.model_selection import RandomizedSearchCV;
import numpy as np;
import traceback;

class explainer:
    '''
    This is a class which encapsulates all kinds of explainers based on SHAP values of a model.

    Author
    -----------
    Kelvin Mock
    '''
    treeExplainer: shap.TreeExplainer = None; # for decision tree or random forest
    linearExplainer: shap.LinearExplainer = None; # for SVM
    permutationExplainer: shap.PermutationExplainer = None; # for MLP
    kernelExplainer: shap.SamplingExplainer = None; # for others like KNN
    model: DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier | KNeighborsClassifier | RandomizedSearchCV;

    def __init__(self, model: DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier | KNeighborsClassifier | RandomizedSearchCV, data: np.ndarray, modelType: str = ""):
        '''
        This is a constructor class which instantiates a suitable explainer based on the model's type.

        Parameters
        ---------------
        model : DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier | KNeighborsClassifier | RandomizedSearchCV - It allows a raw model or a model which is optimized by Randomized Search.
        data : numpy.ndarray - Data array used to instantiate an explainer (usually the samples of the training set, unless too large).
        modelType : str - A string representation of a model's type {"tree", "svm", "neural"}
        '''
        print("Start instantiating an explainer.");
        self.model = model;
        try:
            match (modelType):
                case "tree":
                    self.treeExplainer = shap.TreeExplainer(
                        model=self.model,
                        data=data
                    );
                    print("A tree explainer is instantiated successfully.");
                case "svm":
                    self.linearExplainer = shap.LinearExplainer(
                        model=self.model,
                        data=data,
                        masker=shap.maskers.Partition(data)
                    );
                    print("A Linear explainer is instantiated successfully.");
                case "neural":
                    self.permutationExplainer = shap.PermutationExplainer(
                        model=self.model.predict,
                        data=data,
                        masker=shap.maskers.Partition(data)
                    );
                    print("A Permutation Kernel explainer is instantiated successfully.");
                case _:
                    self.kernelExplainer = shap.SamplingExplainer(
                        model=self.model.predict,
                        data=data
                    );
                    print("A Sampling Kernel explainer is instantiated successfully.");
        except:
            traceback.print_exc();

    def explain(self, X_test: np.ndarray):
        '''
        This is a function which provides explanation based on a SHAP value.

        Parameters
        ------------
        X_test : numpy.ndarray - The data used in an explanation (usually the samples of the test set).

        Returns
        ------------
        {numpy.ndarray} - Estimated SHAP values
        '''
        if (self.treeExplainer != None):
            print("A tree explainer is found.");
            return self.treeExplainer.shap_values(X_test);
        elif (self.linearExplainer != None):
            print("A Linear explainer is found");
            return self.linearExplainer.shap_values(X_test);
        elif (self.permutationExplainer != None):
            print("A Permutation Explainer is found");
            return self.permutationExplainer.shap_values(X_test);
        elif (self.kernelExplainer != None):
            print("A Sampling Kernel explainer is found");
            return self.kernelExplainer.shap_values(X_test);
        else:
            raise ValueError("Missing or Invalid Type of explainer! Please instantiate again.");

    def calBaseVal(self, model: MLPClassifier = None, X_train: np.ndarray = np.array([])):
        '''
        This is a function which calculates the base value (for plotting purpose).

        Parameters
        --------------
        model : MLPClassifier
        X_train : numpy.ndarray

        Returns
        --------------
        numpy.array([float])
        '''
        if (self.treeExplainer != None):
            print("A tree explainer is found.");
            baseVal = self.treeExplainer.expected_value;
        elif (self.linearExplainer != None):
            print("A Linear explainer is found");
            baseVal = self.linearExplainer.expected_value;
        elif (self.permutationExplainer != None):
            print("A Permutation Explainer is found");
            if (len(X_train) == 0 or model == None):
                raise ValueError("Missing model or training data. Cannot evaluate the base value.");
            y_pred_train = model.predict(X_train);
            baseVal = y_pred_train.mean();
        elif (self.kernelExplainer != None):
            print("A Sampling Kernel explainer is found");
            baseVal = self.kernelExplainer.expected_value;
        else:
            raise ValueError("Missing or Invalid Type of explainer! Please instantiate again.");

        return np.array(baseVal);
