import sys

## data transformation related libraries
import pandas as pd
import numpy as np

from sqlalchemy import create_engine ## SQL database module

import argparse ## CLI argument parser library

def load_data(messages_filepath, categories_filepath):
    """ To get the raw data from two csv files one contains messages and another one all the labels to categorize 
        the paticular message both have unique column called id
    Args:
        messages_filepath : CSV file path for messages 
        categories_filepath : CSV file path for all the labeled categories 
    Returns:
        merged dataframe of messages and categories dataframes
    """
    
    disa_cate_df = pd.read_csv(categories_filepath)  ## reading messages data
    disa_msg_df = pd.read_csv(messages_filepath) ## reading categories data
    
    merge_df = disa_cate_df.merge(disa_msg_df) ## merging messages and categories data
    
    return merge_df


def clean_data(df):
    """ To clean the data , which includes transform the categories data into different columns 
        and also assign the particular column names, also drops the duplicate rows from the cleaned data
    Args:
        df : dataframe
    Returns:
        cleaned dataframe
    """
    
    categories_df = df.categories.str.split(";",expand=True) ## creating categories dataframe containing all the different categories as different columns
    category_names = [x.split("-")[0] for x in categories_df.head(1).values.squeeze().tolist()] ## getting category names
    categories_df.columns = category_names
    categories_df = categories_df.applymap(lambda x : int(x.split("-")[-1]))  ## filtering 0s and 1s from all the values
    
    df = pd.concat([df,categories_df],axis=1).drop("categories",axis=1) ## drop actual categories column
    
    df = df.drop_duplicates() ## drop duplicate rows
    
    return df


def save_data(df, database_filename):
    """ Save the cleaned data as a SQL table in sqlite database
    Args:
        df : Cleaned data
        database_filename : database filepath
    Returns:
        Nothing
    """
    
    table_name = "disaster_messages"
    engine = create_engine(f"sqlite:///{database_filename}")   ## creating sqlite engine
    
    df.to_sql(table_name, engine, index=False, if_exists='replace') ## saving sql table to the sqlite database


def main():
    """ 
    main method to run the whole training process
    """
    parser = argparse.ArgumentParser(description="""Please provide the filepaths of the messages and categories 
                                                    datasets as the first and second argument respectively, as 
                                                    well as the filepath of the database to save the cleaned data 
                                                    to as the third argument. Example: python process_data.py 
                                                    disaster_messages.csv disaster_categories.csv DisasterResponse.db """)
    parser.add_argument('messages_filepath', type=str, help='Please ! Enter the database file path')
    parser.add_argument('categories_filepath', type=str, help='Please ! Enter the model filepath')
    parser.add_argument('database_filepath', type=str, help='Please ! Enter the database file path')

    args = parser.parse_args()
    vars_data = vars(args)

    messages_filepath = vars_data["messages_filepath"]
    categories_filepath = vars_data["categories_filepath"]
    database_filepath = vars_data["database_filepath"]

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
    
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
    
    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()