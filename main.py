import os
import sys
sys.path.append(".")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,precision_score,recall_score, classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler,Normalizer

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from importlib import reload
pd.set_option('display.max_columns', 30) 
import seaborn as sns
import pickle
from copy import deepcopy

def description(x):
    t = type(x[0])
    levels = list(x.value_counts().index)
    nbr_levels = len(levels)
    if nbr_levels > 20 : 
        print(f" --- {x.name} --- :{t} | {nbr_levels} \n", *levels[:10], sep=' | ')
        return {"t":t,"nbr_levels":nbr_levels, "levels":levels[:10]}
        
    else:
        print(f" --- {x.name} --- :{t} | {nbr_levels} \n", *levels, sep=' | ')
        return {"t":t,"nbr_levels":nbr_levels, "levels":levels}
def check_na(df, column):
    return pd.DataFrame(df[column].isna().value_counts(normalize=True)*100).T
def dollar_column(df, column):
    df[column] = df[column].replace({'\$': '', '[,]': ''}, regex=True).astype(float)
    return df

def typo_column(df, column):
    df[column] = df[column].apply(lambda x:x.str.lower().replace({'<':'', 
                                                  '/':' ',
                                                  " ":"_",
                                                  'z_':'',
                                                 }, regex=True))
    return df
def save_(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))
    
def load_(filename):
    return pickle.load(open(filename, 'rb'))

class FormatCleaner():
    def __init__(self, dollar_columns,typo_columns):
        self.dollar_columns=dollar_columns
        self.typo_columns=typo_columns
        pass
    
    def clean(self, X):
        X = deepcopy(X)
        X = (X.pipe(dollar_column, dollar_columns)
                .pipe(typo_column, typo_columns)
               )
        return X
    
    def fit_transform(self, X):
        return self.clean(X)

class Transformer():
    def __init__(self, categorical_columns, drop_columns):
        self.drop_columns =  drop_columns
        self.features = []
        self.categorical_columns = categorical_columns
        pass
    
    def fit_transform(self, X) :
        X.drop(self.drop_columns, axis=1,inplace=True)
        X = pd.get_dummies(X, columns =self.categorical_columns, drop_first=True)       
        self.features = X.columns.values
        return X
    
    def transform(self, X):
        X.drop(self.drop_columns, axis=1,inplace=True)
        X = pd.get_dummies(X, columns =self.categorical_columns, drop_first=True) 
        X = X.reindex(columns = self.features, fill_value=0)
        return X
    
class Preprocessor():
    def __init__(self, dollar_columns, typo_columns, categorical_columns, drop_columns):
        self.cleaner =  FormatCleaner(dollar_columns,dollar_columns)
        self.imputer= SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.transformer=Transformer(categorical_columns=categorical_columns, drop_columns=drop_columns )
        self.scaler= MinMaxScaler()
        self.normalizer =Normalizer()
        self.features = []

        pass
    
    def fit_transform(self, X):
        X= self.cleaner.fit_transform(X)
        cols = X.columns.values
        X_imputed = self.imputer.fit_transform(X.values)
        X = self.transformer.fit_transform(pd.DataFrame(X_imputed, columns=cols))
        self.features = self.transformer.features

        X = self.scaler.fit_transform(X.values)
        X = self.normalizer.fit_transform(X)
        return X
    
    
    def transform(self, X):
        X = self.cleaner.fit_transform(X)
        X_imputed = self.imputer.transform(X.values)
        cols = X.columns.values
        X = self.transformer.transform(pd.DataFrame(X_imputed, columns=cols))
        X = self.scaler.transform(X.values)
        X = self.normalizer.transform(X)
        return X
        
def load_data(data_filename):
    """load data set from .csv file as a pd.DataFrame object"""
    data_directory = os.getcwd()
    path=os.path.join(data_directory, "data",data_filename)
    df = pd.read_csv(path)
    print(" --- Data loaded from ", path)
    return df
    
    
    
    
def score_classifier(dataset, classifier, labels):
    predicted_labels = classifier.predict(dataset)
    cls_report =classification_report(labels, predicted_labels, zero_division=0)
    print(cls_report)
    return classifier
    
    
if __name__ == "__main__":
    TRAIN_FILENAME = "train_auto.csv"
    TEST_FILENAME = "test_auto.csv"
    categorical_columns = ['PARENT1', 'MSTATUS', 'SEX', 'CAR_USE', 'RED_CAR', 'REVOKED', 'URBANICITY','CAR_TYPE','JOB','EDUCATION']



    # cleaning typos and logical input
    dollar_columns = ['HOME_VAL','INCOME','BLUEBOOK','OLDCLAIM']
    typo_columns =['PARENT1', 'MSTATUS', 'SEX', 'CAR_USE', 'RED_CAR', 'REVOKED', 'URBANICITY','CAR_TYPE','JOB','EDUCATION']

    # changing types
    cat_to_binary = ['PARENT1', 'MSTATUS', 'SEX', 'CAR_USE', 'RED_CAR', 'REVOKED', 'URBANICITY']
    categorical_to_ordinal =['EDUCATION']

    # read test_auto.csv
    df_test =  load_data(TEST_FILENAME)
    df_train = load_data(TRAIN_FILENAME)

    y_train = df_train['TARGET_FLAG']



    # load pre-processor
    preprocessor = load_("./models/Preprocessor")
    best_model = load_("./models/model_SVC")



    # preprocess data
    X_train = preprocessor.transform(df_train)
    X_test = preprocessor.transform(df_test)

    # evaluate on train set and return scores : most important to look for is recall
    print(" --- Important metric is RECALL of class 1")
    score_classifier(dataset=X_train, 
                                classifier=best_model,
                                labels=y_train)

    # make predictions
    y_pred = best_model.predict(X_test)

    # save to a new file 
    print(" --- Predicting ..")
    pd.DataFrame({"INDEX":df_test['INDEX'], "TARGET_FLAG":y_pred}).to_csv('./prediction_on_auto_test.csv', index=False)  
    print(" --- Predictions saved to ./prediction_on_auto_test.csv")

