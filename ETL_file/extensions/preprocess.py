import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from numpy import nan
from sklearn.preprocessing import PowerTransformer
import dask
import dask_ml
import dask.dataframe as dd
from dask import compute
from dask_ml.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from dask_ml.decomposition import PCA


class Preprocessor():

    """
    A class that compiles several functions that preprocess both numeric and categorical data for a general dataset. 

    Includes functions that handle duplicates, impute/drop missing values and scale/normalize the data.

    It relies on 'dask' library for parallel computing and as a result, can handle large datasets.
    """

    def __init__(self, numeric_features, target_feature, train_dataframe, labels, test_dataframe = None,
                 categorical_features = None, drop_threshold = 50, category_threshold = 1, missing_method = 'fill',
                 scale_method = 'standardscaler', scale_range = (0, 1), transform_method = 'yeo-johnson'):

        """
        Initializes an instance of the Preprocessor class.

        Parameters
        ----------

        numeric_features : list of strings
            A list containing the column labels corresponding to numeric features,
        categorical features: list of strings, default = None
            A list containing the column labels corresponding to categorical features
        target_feature: string
            Column label corresponding to the target feature
        train_df: a Dask Dataframe
            A dataframe containing training data
        test_df: a Dask Dataframe, default = None
            A dataframe containing the testing data
        drop_threshold: int, default = 50
            Percentage of missing values in a feature below which a column will be dropped
        labels: list of strings
            A list containing the expected labels in the target feature
        category_threshold: integer
            Count of categories below which (or equal) a category will be grouped together as 'Other' for a categorical feature
        missing_method: string
            How to handle missing values in a feature, options are 'fill' and 'drop'
                fill: the missing values are imputed with the mean for the numeric features and the mode for categorical features
                drop: the missing values are dropped entirely for both numeric and categorical features
        scale_method: string, default = 'standardscaler'
            Which scaling algorithm to use on the data
                standardscaler: applies Scikit-Learn's StandardScaler; data is centered by subtracting the mean and scaled by dividing by 
                                the standard deviation
                minmax: applies Scikit-Learn's MinMaxScaler; data is scaled to a range specified by the feature_range parameter.
                        Calculation: [(Observation - Observation(minimum)/ (Observation(maximum) - Observation (minimum)))]*(upper_limit - 
                        lower_limit) + lower_limit
                robust: applies Scikit-Learn's RobustScaler; data is centered by subtracting the median and scaled by dividing by the
                        interquartile range. Used for data with outliers.
        scale_range: tuple(lower_limit, upper_limit)
            Specifies the upper and lower limits of scaling; applicable only for the minmax method only.
        transform_method: string, default = 'yeo-johnson'
            Which transformation algorithm to use on the data
                yeo-johnson: works for both positive and negative values
                box-cox: works for positive values only
        """

        self.target_feature = target_feature
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.train_df = train_dataframe
        self.labels = labels
        self.test_df = test_dataframe
        self.drop_threshold = drop_threshold
        self.category_threshold = category_threshold
        self.missing_method = missing_method
        self.scale_method = scale_method
        self.scale_range = scale_range
        self.transform_method = transform_method
        self.scaler = None
        self.transformer = None

        print('Preprocessor ready...')


    def remove_invalid(self, df):

        """
        Method that removes invalid observations. Examples of invalid observations are string characters in
        numeric values, as well as unexpected target values

        Parameters
        ----------
        df: Dask dataframe
            A dataframe for which invalid observations must be removed
        
        Returns
        -------
        df: Dask dataframe
            A dataframe for which invalid observations are removed
        """

        for num in self.numeric_features:
            df[num] = dd.to_numeric(df[num], errors='coerce')
        df= df[df[self.target_feature].isin(list((map(str, self.labels))))]
        return df

    def remove_duplicates(self, df):

        """
        Method that finds AND removes duplicated observations in a dataframe

        Parameters
        ----------
        df : Dask dataframe
            A dataframe for which duplicates must be removed

        Returns
        ----------
        df : Dask dataframe
            A dataframe with duplicates, if any, are removed.
        """

        df = df.drop_duplicates()
        self.df = df
        return df

    def encode_target(self, df):

        """
        Method that handles the preprocessing of the target feature

        Parameters
        ----------
        df : Dask dataframe
            A dataframe for which the target feature must be encoded

        Returns
        ----------
        df : Dask dataframe
            A dataframe with the target feature encoded
        """
        enc = dask_ml.preprocessing.LabelEncoder()
        df[self.target_feature] = enc.fit_transform(df[self.target_feature])
        self.df = df
        return df

    def remove_missing_values(self, df):

        """
        Method that handles missing data in numeric and categorical features

        Parameters
        ----------
        df : Dask dataframe
            A dataframe for which the target feature must be encoded


        Returns
        ----------
        df : Dask dataframe
            A dataframe with the missing values imputed or dropped
        """

        if self.missing_method not in ['fill', 'drop']:
            raise ValueError("method must be one of: fill, drop")

        if self.drop_threshold!=None:
            try:
                int(self.drop_threshold)
            except:
                raise TypeError("Threshold value must be numeric or None")

        if isinstance(self.category_threshold, int):
            pass
        else:
            raise TypeError('Threshold value must be an integer')

        missing = df.isnull().sum()
        if missing.any().compute():
            print("Missing values detected")
            if self.missing_method == 'drop':
                df = df.dropna()

            elif self.missing_method == 'fill':
                if self.drop_threshold!=None:
                    percent_missing = ((missing/df.index.size)*100).compute()
                    drop_list = list(percent_missing[percent_missing > self.drop_threshold].index)
                    df = df.drop(drop_list, axis = 1)
                    self.numeric_features = [i for i in self.numeric_features if i not in drop_list]
                    if self.categorical_features != None:
                        self.categorical_features = [i for i in self.categorical_features if i not in drop_list]
                #NUMERIC COLUMNS
                if df[self.numeric_features].isnull().sum().any().compute():
                    print("Handling missing values from numeric columns...")
                    means = df[self.numeric_features].mean().compute()
                    df[self.numeric_features] = df[self.numeric_features].fillna(means)

                #CATEGORICAL COLUMNS
                if self.categorical_features != None:
                    print("Handling missing values from Categorical columns...")
                    category_count = None
                    if df[self.categorical_features].isna().sum().any().compute():
                        for i in list(self.categorical_features):
                            category_count = df[i].value_counts().compute()
                            mode = category_count.index[0]
                            df[i] = df[i].fillna(mode)
                    if category_count==None:
                        for i in list(self.categorical_features):
                            category_count = df[i].value_counts().compute()
                            distinct = list(category_count[category_count <= self.category_threshold].index)
                            df[i] = df[i].replace(distinct, 'Other')
        self.df = df
        return df

    def scale_data(self, df):

        """
        Method that handles missing data in numeric and categorical features

        Parameters
        ----------
        df : Dask dataframe
            A dataframe for which the target feature must be encoded

        Returns
        ----------
        df : Dask dataframe
            A dataframe that has been scaled

        """

        if self.scale_range[0] >= self.scale_range[1]:
            raise ValueError("Lower limit must be less than upper limit for scaling.")
        if self.scale_method not in ['standardscaler', 'minmax', 'robust']:
            raise ValueError("method must be one of: standardscaler, minmax, or robust")

        if self.scaler == None:
            if self.scale_method == 'standardscaler':
                scaler = dask_ml.preprocessing.StandardScaler()
            elif self.scale_method == 'minmax':
                scaler = dask_ml.preprocessing.MinMaxScaler(feature_range = self.scale_range)
            elif self.scale_method == 'robust':
                scaler = dask_ml.preprocessing.RobustScaler()
            df[self.numeric_features] = scaler.fit_transform(df[self.numeric_features])
            self.scaler = scaler
        else:
            df[self.numeric_features] = self.scaler.transform(df[self.numeric_features])

        return df

    def transform_data(self, df):

        """
        Method that handles transforming data to more Gaussian-like distributions and stabilized variance using scikit-learn PowerTransformer.
        Please use the class's 'scale' method if you wish to apply scaling before using this transformer

        Parameters
        ----------
        df : Dask dataframe
            A dataframe for which the target feature must be encoded

        Returns
        ----------
        df : Dask dataframe
            A dataframe for which numeric features have been transformed to Gaussian-like distributions
            and stabilized variance
        """

        if self.transform_method not in ['yeo-johnson', 'box-cox']:
            raise ValueError("method must be one of: yeo-johnson, box-cox")

        df = df.compute()
        if self.transformer == None:
            transformer = sklearn.preprocessing.PowerTransformer(method = self.transform_method, standardize = False)
            df[self.numeric_features] = transformer.fit_transform(df[self.numeric_features])
            self.transformer = transformer
        else:
            df[self.numeric_features] = self.transformer.fit_transform(df[self.numeric_features])
        return df

    def execute(self, invalid=True, duplicates=True, missing=True, scale=True, transform=True, encode_target=False, train=True):

        """
        Method that executes all the class methods in sequence.

        Parameters
        ----------
        dataframe : Dask dataframe
            A dataframe for which the target feature must be encoded
        duplicates: bool, default = True
            Whether to handle duplicate observations or not
        missing: bool, default = True
            Whether to handle missing values or not
        scale: bool, default = True
            Whether to handle scaling values or not
        transform: bool, default = True
            Whether to handle transforming values values or not
        encode_target: bool, default = True
            Whether to handle encoding the target feature or not
        train: bool, default = True
            Whether training data is being handled, if false, testing data is assumed to be handled.

        Returns
        -------
        df: Dask Dataframe
            A preprocessed dataframe
        """
        if train:
            temp = self.train_df.copy()
        else:
            temp = self.test_df.copy()
        
        if invalid:
            print("Removing invalid observations in dataframe...")
            temp = self.remove_invalid(temp)
        if duplicates:
            print("Removing duplicates in dataframe...")
            temp = self.remove_duplicates(temp)
        if missing:
            print("Handling missing values...")
            temp = self.remove_missing_values(temp)
        if encode_target:
            print("Encoding target feature...")
            temp = self.encode_target(temp)
        if scale:
            print("Scaling data...")
            temp = self.scale_data(temp)
        if transform:
            print("Transforming the numeric features to Gaussian-like distributions and stabilized variance...")
            temp = self.transform_data(temp)
            if train:
                print('Train dataframe has shape: ', temp.shape)
            else:
                print('Test dataframe has shape: ', temp.shape)
            return temp

        df = temp.compute()
        if train:
            print('Train dataframe has shape: ', df.shape)
        else:
            print('Test dataframe has shape: ', df.shape)
        return df
