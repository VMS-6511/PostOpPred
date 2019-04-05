#Author: Vinith Menon Suriyakumar, Christina Yan, Mike Kennelly
#NetID: 13vms1, 14cy5, 13mwjk
#Course: CISC 452, Winter 2019

from scipy.io import arff
from io import StringIO
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
import pandas as pd

#This loads the file containing the data from the UCI Machine Learning repository
data_file = '../data/ThoraricSurgery.arff'
data, meta = arff.loadarff(data_file)

outcomes_df = pd.DataFrame.from_records(data)
nominal_features = ['DGN', 'PRE6',  'PRE14']
TF_features = ['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE19',
              'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']
numeric_features = ['PRE4', 'PRE5', 'AGE']
nominal_dict = {feature: [] for feature in nominal_features}
TF_dict = {feature: [] for feature in TF_features}

# This collects all of the categorical features and true/false features so that they can be preprocessed
for feature in nominal_features:
    nominal_dict[feature][:] = (pd.Series(outcomes_df[feature].values).unique().tolist())
for feature in TF_features:
    TF_dict[feature][:] = (pd.Series(outcomes_df[feature].values).unique().tolist())

#This for loop performs two of the preprocessing steps in our pipeline which are: one hot encoding
# and conversion of true/false strings to 0 and 1 respectively
for feature in outcomes_df.columns.values:
    new_column = []
    # This loop converts the categorical features into one hot encodings
    for i, row in outcomes_df.iterrows():
        if feature in nominal_features:
            encoding = np.zeros(len(nominal_dict[feature])).tolist()
            index = nominal_dict[feature].index(row[feature])
            encoding[index] = 1
            new_column.append(str(encoding))
        # If the feature is a true/false feature then it is converted to a binary value
        if feature in TF_features:
            outcomes_df.loc[i,feature] = TF_dict[feature].index(row[feature])
    if feature in nominal_features:
        outcomes_df.drop([feature], axis=1)
        outcomes_df[feature] = pd.DataFrame(new_column, columns=[feature])
        
# These functions sepsrate the dataset into the two classes so that the imbalance within the dataset
# can be observed and used for the following upsampling
Risk1Yr_1_df = outcomes_df.loc[outcomes_df['Risk1Yr'] == 1]
Risk1Yr_0_df = outcomes_df.loc[outcomes_df['Risk1Yr'] == 0]

# This for loop upsamples the patients who died within 1 year of their procedures since
# the dataset had a class imbalance and results in an approximate 1:1 ratio of the dataset
for i in range(4):
    outcomes_df = outcomes_df.append(Risk1Yr_1_df)

# Using Scikit-learn we split the dataset into an 80-20 training and test split that uses stratification to ensure
# that the proportion of classes in the dataset is maintained in both the training and test sets
outcomes_train, outcomes_test = train_test_split(outcomes_df, test_size=0.2, stratify=outcomes_df['Risk1Yr'].values.tolist())
outcomes_train_df = pd.DataFrame(outcomes_train)
outcomes_test_df = pd.DataFrame(outcomes_test)
print(outcomes_train_df.head(5))
# This function is used for standardizing the features to reduce the variance and improve the neural network
# performance
def standardization(x, mean, std):
    z_scores_np = (x - mean) / std
    return z_scores_np

# This function rescales the features using the standardization function for the training set
def train_standard_loop(keys):
    for key in keys:
        x_np = outcomes_train_df[key].values
        x_np_mean = x_np.mean()
        x_np_std = x_np.std()
        outcomes_train_df[key] = outcomes_train_df[key].apply(standardization,args=(x_np_mean,x_np_std))

# This function rescales the features using the standardization function for the test set
def test_standard_loop(keys):
    for key in keys[1:10]:
        x_np = outcomes_test_df[key].values
        x_np_mean = x_np.mean()
        x_np_std = x_np.std()
        outcomes_test_df[key] = outcomes_test_df[key].apply(standardization,args=(x_np_mean,x_np_std))

train_standard_loop(list(['PRE4', 'PRE5', 'AGE']))
test_standard_loop(list(['PRE4', 'PRE5', 'AGE']))

outcomes_train_df.to_csv(path_or_buf='outcomes_data_train.csv',index=False)
outcomes_test_df.to_csv(path_or_buf='outcomes_data_test.csv',index=False)

