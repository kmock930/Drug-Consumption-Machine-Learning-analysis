import shap;
import shap.maskers
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier;
from sklearn.svm import SVC;
from sklearn.neural_network import MLPClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.model_selection import RandomizedSearchCV;
import numpy as np;
import constants;
import traceback;

class explainer:
    treeExplainer: shap.TreeExplainer = None; # for decision tree or random forest
    linearExplainer: shap.LinearExplainer = None; # for SVM
    kernelExplainer: shap.SamplingExplainer = None; # for MLP
    model: DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier | KNeighborsClassifier | RandomizedSearchCV;

    def __init__(self, model: DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier | KNeighborsClassifier | RandomizedSearchCV, data: np.ndarray, modelType: str = ""):
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
                case "mlp":
                    self.kernelExplainer = shap.SamplingExplainer(
                        model=self.model.predict,
                        data=data
                    );
                    print("A Sampling Kernel explainer is instantiated successfully.");
                case "":
                    raise ValueError("Missing or Invalid Type of explainer! Please instantiate again.");
        except:
            traceback.print_exc();

    def explain(self, X_test: np.ndarray):
        if (self.treeExplainer != None):
            print("A tree explainer is found.");
            return self.treeExplainer.shap_values(X_test);
        elif (self.linearExplainer != None):
            print("A Linear explainer is found");
            return self.linearExplainer.shap_values(X_test);
        elif (self.kernelExplainer != None):
            print("A kernel explainer is found");
            return self.kernelExplainer.shap_values(X_test);
        else:
            raise ValueError("Missing or Invalid Type of explainer! Please instantiate again.");