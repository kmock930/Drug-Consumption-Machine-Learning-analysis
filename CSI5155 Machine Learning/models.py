import numpy as np;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn import svm;
import joblib;
import constants;

'''
@summary A class which encapsulates various models which have been initialized
'''
class Models:
    # classifiers
    decisionTree_clf = None;
    randomForest_clf = None;
    svm_clf = None;

    def __init__(self):
        self.decisionTree_clf = DecisionTreeClassifier(
            criterion=constants.entropyCriterion,
            splitter=constants.splitter,
            max_depth=None,
            random_state=np.random.RandomState
        );
    
        self.randomForest_clf = RandomForestClassifier(
            criterion=constants.entropyCriterion,
            bootstrap=True, # the dataset is split into different trees
            random_state=np.random.RandomState
        );

    '''
    @param args: pass as many string as possible to specify which model to output
    Possible models:
    - 'decision tree'
    - 'random forest'
    '''
    def getModels(self, *args):
        res = {};
        if (args == {} or constants.descisionTree in args):
            res.update({constants.descisionTree: self.decisionTree_clf});
        if (args == {} or constants.randForest in args):
            res.update({constants.randForest: self.randomForest_clf});
        
        return res

    def saveModels(self, isTrained=False, **args):
        try:
            # format part of the filename
            # by deciding whether the saved model file is trained or untrained
            isTrained_string = "";
            if (isTrained):
                isTrained_string = '_posttrained';
            else:
                isTrained_string = '_pretrained';
            
            if (args == {} or constants.descisionTree in args):
                joblib.dump(self.decisionTree_clf, 'model_decisionTree' + isTrained_string + '.pkl');
            if (args == {} or constants.randForest in args):
                joblib.dump(self.randomForest_clf, 'model_randomForest' + isTrained_string + '.pkl');
        except:
            return False;
        return True;