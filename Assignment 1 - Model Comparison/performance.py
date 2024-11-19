import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import sklearn.metrics as metrics;
from sklearn.metrics import roc_curve;
import constants;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier;
from sklearn.svm import SVC;
from sklearn.neural_network import MLPClassifier;
from sklearn.neighbors import KNeighborsClassifier;
import seaborn as sns;

'''
@summary Function that evaluates number of True Positives
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@trueLabel strung: a string representation of the true label
@return int: the total count
'''
def evalTP(y_pred, y_test, trueLabel):
    count = 0
    for i in range(y_test.size):
        actual = y_test[i];
        pred = y_pred[i];
        if (actual == pred and pred == trueLabel):
            # falsely predicted and the predicted label is false (negative)
            count += 1;
    return count;

'''
@summary Function that evaluates the number of False Negatives
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@trueLabel strung: a string representation of the true label
@return int: the total count
'''
def evalFN(y_pred, y_test, trueLabel):
    count = 0
    for i in range(y_test.size):
        actual = y_test[i];
        pred = y_pred[i];
        if (actual != pred and pred != trueLabel):
            # falsely predicted and the predicted label is false (negative)
            count += 1;
    return count;

'''
@summary Function that evaluates the number of False Positives
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@trueLabel strung: a string representation of the true label
@return int: the total count
'''
def evalFP(y_pred, y_test, trueLabel):
    count = 0
    for i in range(y_test.size):
        actual = y_test[i];
        pred = y_pred[i];
        if (actual != pred and pred == trueLabel):
           # falsely predicted and the predicted label is false (negative)
            count += 1;
    return count;

'''
@summary Function that evaluates the number of true negatives
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@trueLabel strung: a string representation of the true label
@return int: the total count
'''
def evalTN(y_pred, y_test, trueClass):
    count = 0
    for i in range(y_test.size):
        actual = y_test[i];
        pred = y_pred[i];
        if (actual == pred and pred != trueClass):
            # falsely predicted and the predicted label is false (negative)
            count += 1;
    return count;


'''
@summary Function that evaluates performance metrics
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@return float: accuracy score
'''
def evalAccuracy(y_pred, y_test):
    # Evaluate an array of boolean by comparing each entry from the predicted labels vs the actual labels, so that we know the overall accuracy of the model.
    evalArr = y_pred == y_test;
    
    correct_pred = 0;
    for boolItem in evalArr:
        if (boolItem == True):
            correct_pred += 1;
    return correct_pred / y_test.size;

def evalPrecision(y_pred, y_test):
    tp = evalTP(y_pred, y_test, 1);
    fp = evalFP(y_pred, y_test, 1);
    predPos = tp + fp;
    try:
        return tp / predPos;
    except ZeroDivisionError:
        return 0.0;

def evalRecall(y_pred, y_test):
    tp = evalTP(y_pred, y_test, 1);
    fn = evalFN(y_pred, y_test, 1);
    actualPos = tp + fn;
    try:
        return tp / actualPos;
    except ZeroDivisionError:
        return 0.0;

def evalF1Score(y_pred, y_test):
    precision = evalPrecision(y_pred, y_test);
    recall = evalRecall(y_pred, y_test);
    try:
        return 2 * (precision * recall) / (precision + recall);
    except ZeroDivisionError:
        return 0.0;

'''
@summary Function that displays the prediction performance in a well-formatted confusion matrix
@param y_pred numpy.array: an array of predicted labels
@param y_test numpy.array: an array of actual labels
@return pandas.core.frame.DataFrame The resulting confusion matrix in a Pandas DataFrame format
'''
def printConfMtx(y_pred, y_test):
    pandas_y_actual = pd.Series(y_test, name='Actual');
    pandas_y_pred = pd.Series(y_pred, name='Predicted');
    confMtx = pd.crosstab(
        pandas_y_actual, 
        pandas_y_pred, 
        rownames=['Actual'], 
        colnames=['Predicted'], 
        margins=True);
    confMtx.rename(
            index={0: 'non-user', 1: 'user'},
            columns={0: 'non-user', 1: 'user'}, 
            inplace=True
        );
    new_row_order = ['user', 'non-user', 'All']  # Define your desired row order
    new_column_order = ['user', 'non-user', 'All']  # Define your desired column order
    confMtx = confMtx.reindex(index=new_row_order, columns=new_column_order); # reverse order of col and row
    confMtx.fillna(0, inplace=True);
    return confMtx;

def evalPredictionNum(y_pred: np.ndarray, y_test: np.ndarray, truePred: bool = None):
    if (y_test is None):
        raise ValueError("Please specify the actual labels for evaluation.");
    if (y_pred is None):
        raise ValueError("Please specify the predicted labels for evaluation.");
        
    evalArray: np.ndarray = y_pred == y_test;
    if (truePred == None):
        return np.sum(evalArray);  # count of true predictions
    else:
        return np.sum(evalArray == truePred)  # count of true or false predictions

def plotROC(models: dict, X_test: np.ndarray, y_test: np.ndarray | list, dataset: str, feature_subsets: dict = None):
    plt.figure(figsize=(10, 8));
    # https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.plot_roc_curve.html
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {dataset} dataset');

    # Iterate through the models
    AUCs = [];
    if len(models) > 0:
        for modelName, model in models.items():
            if feature_subsets and modelName in feature_subsets:
                X_test_subset = X_test[:, feature_subsets[modelName]];
            else:
                X_test_subset = X_test;
            
            y_scores = getYScore(model, X_test_subset);
            
            fpr, tpr, _ = roc_curve(y_test, y_scores);
            auc = getAUC(y_test, y_scores);
            plt.plot(fpr, tpr, label=f'{modelName} (AUC = {auc:.2f})');
            AUCs.append(auc);

    plt.legend(loc='lower right');
    plt.plot([0, 1], [0, 1], 'r--');
    plt.xlim([0, 1]);
    plt.ylim([0, 1]);
    plt.ylabel('True Positive Rate');
    plt.xlabel('False Positive Rate');
    plt.show();
    return AUCs;

def getAUC(y_test, y_prob_scores):
    return metrics.roc_auc_score(y_test, y_prob_scores);

def getYScore(model, X_test: np.ndarray):
    if isinstance(model, tuple):
        return getYScore(model[0], X_test), getYScore(model[1], X_test);
    # Get predicted probabilities (for classifiers like SVM, set probability=True)
    if hasattr(model, "predict_proba"):
        # Get probabilities for the positive class
        y_scores = model.predict_proba(X_test)[:, 1];
    else:
        # Use decision function for models like SVM without predict_proba
        y_scores = model.decision_function(X_test);
    return y_scores;

# AUC校對圖
def plotAUC_comparison(plotData: dict, dataset: str):
    df = pd.DataFrame(plotData);
    plt.figure(figsize=(10, 6));
    sns.barplot(x='Classifier', y='AUC', hue='Method', data=df);
    plt.title(f'Comparison of AUCs for Classifiers Under Different Balancing Methods - {dataset} dataset');
    plt.xlabel('Classifier');
    plt.ylabel('AUC');
    plt.legend(title='Balancing Method', loc='lower right');
    plt.tight_layout();
    plt.show();