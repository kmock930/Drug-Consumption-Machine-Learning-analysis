import shap;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier;
from sklearn.svm import SVC;
from sklearn.neural_network import MLPClassifier;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.model_selection import RandomizedSearchCV;
import numpy as np;
import traceback;

class explainer:
    treeExplainer: shap.TreeExplainer = None; # for decision tree or random forest
    kernelExplainer: shap.KernelExplainer = None; # for SVM and MLP
    model: DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier | KNeighborsClassifier | RandomizedSearchCV;

    def __init__(self, model: DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier | SVC | MLPClassifier | KNeighborsClassifier | RandomizedSearchCV, X_train: np.ndarray, modelType: str = ""):
        print("Start instantiating an explainer.");
        self.model = model;
        try:
            match (modelType):
                case "tree":
                    self.treeExplainer = shap.TreeExplainer(
                        model=self.model,
                        data=X_train
                    );
                    print("A tree explainer is instantiated successfully.");
                case "svm" | "mlp":
                    self.kernelExplainer = shap.KernelExplainer(
                        model=self.model.predict,
                        data=X_train
                    );
                    print("A kernel explainer is instantiated successfully.");
                case "":
                    raise ValueError("Missing or Invalid Type of explainer! Please instantiate again.");
        except:
            traceback.print_exc();

    def explain(self, X_test: np.ndarray):
        if (self.treeExplainer != None):
            print("A tree explainer is found.");
            return self.treeExplainer.shap_values(X_test);
        elif (self.kernelExplainer != None):
            print("A kernel explainer is found");
            return self.kernelExplainer.shap_values(X_test);
        else:
            raise ValueError("Missing or Invalid Type of explainer! Please instantiate again.");