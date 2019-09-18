import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
# Read in data
housing_data = pd.read_csv('C:/Users/Yibing Qi/Desktop/melbourne-housing-snapshot/melb_data.csv')

# preview & describe the data
print(housing_data.head())
print(housing_data.describe())
print(housing_data.shape)

# data cleaning and preparation
# Drop 2 columns that has too many missing values or useless
housing_data.drop(['Address','CouncilArea'], axis=1)
# Drop rows with missing data
housing_data.dropna(inplace=True, axis=0)
y = housing_data.Price
X = housing_data.drop('Price', axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)
print(X_train.columns)
### Deal with categorical data
### For random forest model, Label Encoding is a decent way to deal with ctgrcl data
# Get categorical cols
ctgrcl_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
# Columns that can be safely label encoded
good_label_cols = [col for col in ctgrcl_cols if
                   set(X_train[col]) == set(X_valid[col])]
print(good_label_cols) #['Type', 'Method', 'Date', 'Regionname']
# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(ctgrcl_cols)-set(good_label_cols))
# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)
# Apply label encoder
label_encoder = LabelEncoder()  # Your code here
for col in good_label_cols:
    label_X_train[col] = label_encoder.fit_transform(label_X_train[col])
    label_X_valid[col] = label_encoder.transform(label_X_valid[col])

### Modeling
my_model = RandomForestRegressor(n_estimators=100, random_state=0)
my_model.fit(label_X_train, y_train)
preds = my_model.predict(label_X_valid)

###validation
# Scatter Plot: Pred vs real
plt.scatter(y_valid, preds, alpha=0.5)
plt.plot([0,8000000],[0,8000000])  # Reference diagonal line
plt.title('Scatter plot Pred vs real')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

tstat,pval = ttest_ind(preds,y_valid)
print(pval) #p_val:0.4127







