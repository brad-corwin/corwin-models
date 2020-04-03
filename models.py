import logging
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import ensemble
from pathlib import Path
from joblib import dump

logger = logging.getLogger(__name__)

class Model(object):
    def __init__(self, df, pred_col, model=None, cat_cols=[], drop_cols=[], date_col=None, time_series=False):
        self.df = df
        self.pred_col = pred_col
        self.cat_cols = cat_cols
        self.drop_cols = drop_cols
        self.date_col = date_col
        self.model = model
        self.time_series = time_series
        self.enc = None
        self.scaler = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.input_columns = None
        
        if date_col is not None:
            drop_cols.append(date_col)
        
    def train_model(self, train, validate):
        raise NotImplementedError
        
    def fit(self):
        # split train and test data
        self.get_train_test()
        
        # fit model
        self.train_model()
    
    def predict(X):
        raise NotImplementedError
    
    # Fits a One Hot Encoder to the categorical columns
    def ohe(self, X):
        # fit OHE
        if self.enc is None:
            self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            Xe = pd.DataFrame(self.enc.fit_transform(np.array(X[self.cat_cols])), index=X.index, columns=self.enc.get_feature_names(self.cat_cols))
        else:
            Xe = pd.DataFrame(self.enc.transform(np.array(X[self.cat_cols])), index=X.index, columns=self.enc.get_feature_names(self.cat_cols))

        # update dataframe with OHE columns
        return X.merge(Xe, left_index=True, right_index=True).drop(self.cat_cols, axis=1)
    
    # Returns the Mean Absolute Error
    def get_mae(self):
        y_pred = self.model.predict(self.X_test)

        return mean_absolute_error(y_pred, self.y_test)
    
    # Returns the Mean Squared Error
    def get_mse(self):
        y_pred = self.model.predict(self.X_test)
        
        return mean_squared_error(y_pred, self.y_test)
    
    # Builds a full prediction dataframe to return
    def get_prediction_df(self, dfi, lagged_periods, forecast_periods, freq):

        if not self.time_series:
            logger.warning('You are trying to build a time-series output with non time-series data. Please set time_series to True if this is time-series data.')
            return
        
        # get the future date range from the latest date
        latest_dt = dfi[self.date_col].max()
        
        if freq == 'W':
            dr = pd.date_range(start=latest_dt+pd.DateOffset(weeks=1), periods=forecast_periods, freq=sfreq)
        elif freq == 'MS':
            dr = pd.date_range(start=latest_dt+pd.DateOffset(months=1), periods=forecast_periods, freq=freq)
        elif freq == 'M':
            dr = pd.date_range(start=latest_dt+MonthEnd(1), periods=forecast_periods, freq=freq)
        
        # call the prediction
        self.predict(dfi, lagged_periods=lagged_periods, forecast_periods=forecast_periods)
        
        # build the dataframe for the future date range
        temp = pd.DataFrame(self.y_pred, index=dr, columns=[self.pred_col]).reset_index().rename({'index': self.date_col}, axis=1)
        temp['predicted'] = True

        # merge the existing data to the forecast data
        temp = pd.concat([temp, dfi])
        temp[self.date_col] = pd.to_datetime(temp[self.date_col], utc=True)
        temp = temp.sort_values(self.date_col).reset_index(drop=True)
        temp['predicted'] = temp['predicted'].fillna(False)
        
        # fill in categorical column information for the predicted columns
        for col in self.cat_cols:
            temp[col].ffill(inplace=True)
        
        return temp
    
    # save the model class to a file
    def save_model(self, file_name, drop_data=True):
        if drop_data:
            self.df = None
            self.X_train = None
            self.y_train = None
            self.X_test = None
            self.y_test = None

        dump(self, Path(file_name))


class SKLearn_Model(Model):
    
    # define and train the model
    def train_model(self):
        # fit model
        self.model.fit(self.X_train, self.y_train)
    
    # split, encode, scale the data to prep for training
    def get_train_test(self, split=0.2, scale=True):
        
        temp = self.df.drop(self.drop_cols, axis=1).dropna()
        
        X = temp.drop([self.pred_col], axis=1)
        y = temp[self.pred_col]

        self.input_columns = X.columns.astype(list)

        if self.cat_cols and self.cat_cols is not None:
            X[self.cat_cols] = X[self.cat_cols].astype(str)
            X = self.ohe(X)
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split)

        if scale:
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
            else:
                self.X_train = self.scaler.transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
        
    # Builds the prediction array for the future values    
    def predict(self, dfi, lagged_periods=None, forecast_periods=None):
        # initial setup
        if self.enc is not None:
            dfi = self.ohe(dfi)
        
        if self.time_series:
            self.y_pred = np.array([])

            pred_arr = np.array(dfi.iloc[-lagged_periods:][self.pred_col])
            ohe_arr = np.array(dfi.drop([self.date_col, self.pred_col], axis=1).iloc[-1])

            for _ in range(forecast_periods):
                # TODO: add error checking in to make sure the array is of the correct size
                current_input = np.append(pred_arr, ohe_arr).reshape(1,-1)
                current_input = self.scaler.transform(current_input)
                pred = self.model.predict(current_input)

                # pop first element
                pred_arr = np.delete(pred_arr, 0)

                # add prediction
                pred_arr = np.append(pred_arr, pred)
                self.y_pred = np.append(self.y_pred, pred)
                
        else:
            self.y_pred = self.model.predict(dfi)

        return self.y_pred

    def get_significant_features(self):
        temp = self.ohe(self.df)

        feature_significance = pd.DataFrame({
            'feature': temp.drop(self.drop_cols, 1).columns,
            'significance': self.model.feature_importances_.round(3)
        })

        # Sum all the categorical variables significance
        for label in self.cat_cols:
            feature_significance.loc[len(feature_significance) + 1] = \
                [label, feature_significance.loc[feature_significance['feature'].str.startswith(label + '_')][
                    'significance'].sum()]

        # Print a list of features and their significance to the prediction
        feature_significance = feature_significance[~feature_significance['feature'].str. \
            startswith(tuple(label + '_' for label in self.cat_cols))]
        feature_significance.reset_index(drop=True)

        return feature_significance.sort_values('significance', ascending=False).set_index('feature')

    def get_significant_features(self, ungrouped=False):
        
        if self.enc is not None:
            feature_significance = pd.DataFrame({
                'feature': self.enc.get_feature_names(self.cat_cols),
                'significance': self.model.feature_importances_.round(3)
            })

            if ungrouped is False:
                # Sum all the categorical variables significance
                for label in self.cat_cols:
                    feature_significance.loc[len(feature_significance) + 1] = \
                        [label, feature_significance.loc[feature_significance['feature'].str.startswith(label + '_')][
                            'significance'].sum()]

                # Print a list of features and their significance to the prediction
                feature_significance = feature_significance[~feature_significance['feature'].str. \
                    startswith(tuple(label + '_' for label in self.cat_cols))]

            feature_significance.reset_index(drop=True)

            return feature_significance.sort_values('significance', ascending=False).set_index('feature')
        
        else:
            # TODO: implement numerical only dataframe
            pass


class RF_Model(SKLearn_Model):
    def train_model(self):
        self.model = ensemble.RandomForestRegressor()
        self.model.fit(self.X_train, self.y_train)