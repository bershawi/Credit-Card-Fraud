import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset from the csv file using pandas
data = pd.read_csv('creditcard.csv')

# Start exploring the dataset
print(data.columns)

# Print The shape of The data
data = data.sample(frac = 0.1,random_state=1)
print(data.shape)

#Plot histogram of each parameter
#data.hist(figsize=(20,20))
#plt.show()

# Detection number of fraud in dataset
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('Fraud cases: {}'.format(len(Fraud)))
print('Valid cases:{}'.format(len(Valid)))

#Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12,9))

sns.heatmap(corrmat,vmax=0.8,square=True)
plt.show()

#Get all columns from the dataframe
columns = data.columns.tolist()

#Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ['Class']]

#Store the variable we'll predict on
target  = 'Class'
X =data[columns]
Y = data[target]

#Print The shape of X and Y
print(X.shape)
print(Y.shape)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#Define a random state
state = 1

#Define The outlier detecion methods
classifiers = {
    'Isolation Forest': IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    'Local Outlier Factor': LocalOutlierFactor(
        n_neighbors=20,
    contamination=outlier_fraction)
}

#Fit The model
n_outliers = len(Fraud)
for i,(clf_name,clf)in enumerate(classifiers.items()):
    #Fit The data and tag the outlier
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        score_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)


    #Reshape The prediction 0 for Valid and 1 for Fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    #Rune classification metrices

    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))