
## Importing Libraries for data transformation
import sys
import pandas as pd 
import numpy as np

from sqlalchemy import create_engine ## SQL database module
import joblib ## save model

## Scikit learn's modules
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer 
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score

## XGboost module
from xgboost import XGBClassifier

import re ## regex library 
import argparse ##cli argument oarser libraries

## Libraries for NLP
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

import time

## Download important modules if possible
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

random_state = 122

def load_data(database_filepath):
    """ To load the data from sqlite database 
    Args:
        database_filepath : Sqlite databse path
    Returns:
        Independant features , target values and category names
    """
    
    engine = create_engine(f"sqlite:///../data/{database_filepath}") ## creating sqlite engine
    df = pd.read_sql_table("disaster_messages", engine ) ## reading sql table 
    
    X = df["message"] ## only using message columns
    Y = df.drop(columns=["id", "message", "original", "genre"]) ## filtering all the category columns
    category_names = Y.columns
    Y = Y.values
    
    return X, Y, category_names


def tokenize(text):
    """ A tokenizer to do the text preprocessing like word tokenizing , removing punctuations , removing stopwords, lemmetiation
    Args:
        text : A sentence
    Returns:
        tokens
    """

    stop_words = stopwords.words("english") ## stop words list
    lemmatizer = WordNetLemmatizer() ## word net lemmatizer object
    
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+" ## url regex
    
    ## replacing urls with static text 
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    ## generating tokens
    tokens = [ lemmatizer.lemmatize(word).strip() for word in word_tokenize(text.lower()) if word.isalnum() and word not in stop_words]
    return tokens


def evaluate_model(model, X_test, Y_test, category_names):
    """ To evaluate the trained model by calculating the classification report and accuracy score for each and aevery labels
    Args:
        model : Trained model
        X_test : Testing data
        Y_test : Testing labels
        category_names : Category/feature names
    Returns:
        Nothing
    """
    
    print(f"Evaluating the model on test data ....")
    
    ## predicting the labels
    Y_pred = model.predict(X_test)
    
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], Y_pred[:, i]))
        print("Accuracy of %25s: %.2f" %(category_names[i], accuracy_score(Y_test[:, i], Y_pred[:,i])))

def build_model(grid_search):
    """ 
    Args:
        grid_search : Boolean Flag 
    Returns:
        Model pipeline 
        
    """
    
    print(f" Using XGBclassifier ...")

    ## classifier object initialization
    classifier = XGBClassifier()

    ## creating pipelines
    nlp_pipeline = Pipeline([ ("count_vect", CountVectorizer(tokenizer=tokenize)),("tf_idf_transf" ,TfidfTransformer() ) ])
    model_pipeline = Pipeline([ ("feature_extractor" , nlp_pipeline) , ("classifier" , MultiOutputClassifier(classifier)) ])

    ## if grid search flag is on then use grid search to get best classifier model
    if grid_search:
        print(f" Grid Search CV is True ...")
        print(f" Finding the best model ...")

        ## all the parameters list from grid search CV
        parameters = {"feature_extractor__count_vect__ngram_range": ((1, 1), (1, 2)) ,
            "feature_extractor__count_vect__max_df": (0.5, 0.75, 1.0) ,
            "feature_extractor__tf_idf_transf__use_idf": (True, False) ,
            "classifier__estimator__learning_rate"    : [0.01, 0.001] ,
            "classifier__estimator__max_depth"        : [ 6, 8, 9],
            "classifier__estimator__min_child_weight" : [ 1, 3, 5 ],
            "classifier__estimator__colsample_bytree" : [ 0.5 , 0.7 ] ,
            "classifier__estimator__n_estimators": [50, 100, 200]
        }

        model_pipeline = GridSearchCV(model_pipeline, param_grid = parameters)
    
    return model_pipeline

def save_model(model, model_filepath):
    """ Save the trained model
    Args:
        model : Trained model
        model_filepath : Range of stress factor
    Returns:
        Nothing
    """
    
    print(f" Saving the model ... ")
    joblib.dump(model, model_filepath)


def main():
    """ 
    main method to run the whole training process
    """
    parser = argparse.ArgumentParser(description=""" Please provide the filepath of the disaster messages database 
                                                as the first argument and the filepath of the pickle file to 
                                                save the model to as the second argument. 
                                                There is also an optional parameter to use grid search CV
                                                Example: python train_classifier.py ../data/DisasterResponse.db classifier.pkl [False/True]""")
    parser.add_argument('database_filepath', type=str, help='Please ! Enter the database file path')
    parser.add_argument('model_filepath', type=str, help='Please ! Enter the model filepath')
    parser.add_argument('--grid_search', type=bool, default=False, help='To run grid search CV or not true/false')

    args = parser.parse_args()
    vars_data = vars(args)

    if len(vars_data.keys()) >= 2:

        database_filepath= vars_data["database_filepath"]
        model_filepath = vars_data["model_filepath"]
        grid_search = vars_data["grid_search"]

        print(" Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
        
        print(" Building model...")
        model = build_model(grid_search)
        
        print(" Training model...")
        model.fit(X_train, Y_train)
        
        print(" Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print(" Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print(" Trained model saved!")

    else:
        print(" Please provide the filepath of the disaster messages database "\
                " as the first argument and the filepath of the pickle file to "\
                " save the model to as the second argument. \n There is also an optional parameter to use grid search CV \n\n Example: python "\
                " train_classifier.py ../data/DisasterResponse.db classifier.pkl [False/True]")


if __name__ == "__main__":
    main()