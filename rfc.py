# Libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

# Globals
file_dir = './data/'
start = time()

# Import train data
train = pd.read_csv('{0}train.csv'.format(file_dir))

# Data wrangling
train.dropna()

# Feature classifiers
class DayOfWeek(BaseEstimator, TransformerMixin):
	def get_feature_names(self):
		return [self.__class__.__name__]

	def fit(self, df, y=None):
		return self

	def transform(self, df):
		return df['DayOfWeek'].as_matrix()[None].T.astype(np.float)

class PdDistrict(BaseEstimator, TransformerMixin):
	def get_feature_names(self):
		return [self.__class__.__name__]

	def fit(self, df, y=None):
		return self

	def transform(self, df):
		return df['PdDistrict'].as_matrix()[None].T.astype(np.float)

def results(df, n=10, column='prediction', merge_column='Category'):
	return ' '.join(df.sort_index(by=column)[-n:])

feature_list = [
	('DayOfWeek', DayOfWeek()),
	('PdDistrict', PdDistrict())
]

# Set up columns for random forest, train
feature_union = FeatureUnion(transformer_list=feature_list)
x = feature_union.fit_transform(train)
y = train['Category']

# Train model
model = RandomForestClassifier(n_jobs=2)
model.fit(x,y)

# Import test data
test = pd.read_csv('{0}test.csv'.format(file_dir))

# Columns for random forest, test
x_test = feature_union.transform(test_df)

# Use model on test dataset
prediction = model.predict_proba(x_test)
pos_idx = np.where(model.classes_ == True)[0][0]
test_df['prediction'] = prediction[:, pos_idx]

# Export submission
submission = test_df.apply(results)
submission.to_csv('submission.csv', header=True)

# Done
print('Finished. Script ran in {0} seconds'.format(time() - start))

