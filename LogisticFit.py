from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#read in data
exp = pd.read_csv('data/expeditions.csv', index_col=0, usecols=[0,2,3,4,13,14,15,16,21,34,35,36,39,42,45,48,49,50])

#drop entries of unrecognised expeditions
exp = exp.drop(exp.index[exp.claimed == True].tolist(), axis='rows')
exp = exp.drop('claimed', axis='columns')

#compile multiple success field into one - count number of successful routes
exp['success'] = exp[['success1', 'success2', 'success3', 'success4']].sum(axis=1).map(lambda x: True if x > 0 else False)
exp = exp.drop(['success1', 'success2', 'success3', 'success4'], axis='columns')

#drop entries where the amount of hired members is missing
exp = exp.drop(exp.index[(exp.tothired == 0) & (exp.nohired == False)].tolist(), axis='rows')
exp = exp.drop('nohired', axis='columns')

#drop entries where the amount of expedition members is set to 0
exp = exp.drop(exp.index[exp.totmembers == 0].tolist(), axis='rows')

#use one-hot encoding on season field
seasons = exp.pop('season')
exp[['dummy_spring', 'dummy_summer', 'dummy_autumn', 'dummy_winter']] = OneHotEncoder(sparse_output=False).fit_transform(seasons.values.reshape(-1,1))

#Add features for quadratic behaviour of totmembers and tothired
exp['totmembers2'] = exp['totmembers']**2
exp['tothired2'] = exp['tothired']**2
exp['totmemberstothired'] = exp['totmembers'] * exp['tothired']

#keep peaks which were attempted more than 5 times, use one-hot-encdoing
view = exp.groupby('peakid').peakid.count()
droppeaks = view.loc[view <= 5].index.values
exp = exp.drop(exp.index[exp.peakid.isin(droppeaks)], axis='rows')

peaks = exp.pop('peakid')
encoder = OneHotEncoder(sparse_output=False)
peaks = encoder.fit_transform(peaks.values.reshape(-1,1))
exp.reset_index(drop=True, inplace=True)
exp = pd.concat([exp, pd.DataFrame(peaks, columns = [entry[3:] for entry in encoder.get_feature_names_out()])], axis=1)

#split target field
Y = exp.pop('success').values
X = exp.values

#scaling
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

#cross validation to find optimal regularization, store coefficients
Cs = [100, 50, 20, 10, 1, 0.5, 0.1]
CV = StratifiedKFold(n_splits = len(Cs), shuffle=True)
scores = {}
coefficients = {}
for i, (train_index, test_index) in enumerate(CV.split(scaled_X, Y)):
    model = LogisticRegression(C = Cs[i]).fit(scaled_X[train_index], Y[train_index])
    pred_Y = model.predict(scaled_X[test_index])
    score = roc_auc_score(Y[test_index], pred_Y)
    #if ROC AUC score for this model is higher than the others, overwrite coefficients
    if (i == 0) or (score > np.max(list(scores.values()))):
        for j in range(len(exp.columns)):
            coefficients[exp.columns.values[j]] = model.coef_[0][j]
        coefficients['intercept'] = model.intercept_[0]
    scores[Cs[i]] = score

for key, value in scores.items():
    print("C: {0:.2f}, score: {1:.3f}".format(key, value))

optC = Cs[np.argmax(list(scores.values()))]
print("Optimal C: {}\n".format(optC))

#print 10 most important fields and their coefficients:
print("Most influential fields and corresponding coefficients:")
for key, value in sorted(coefficients.items(), key=lambda x: np.abs(x[1]), reverse=True)[:10]:
    print('{0}: {1:.2f}'.format(key, value))

#plot dependence of success chance on total amount of expedition members
#we only investigate team sizes up to 30, as the EDA has shown teams exceeding this size to be rare.
X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size = 0.1, shuffle=True)
model = LogisticRegression(C = optC).fit(X_train, Y_train)
modedata = exp.mode().iloc[0]
datagrid = np.zeros((len(modedata), 30, 31))
for i in range(30):
    for j in range(31):
        modedata['totmembers'] = i+1
        modedata['tothired'] = j
        modedata['totmembers2'] = modedata['totmembers']**2
        modedata['totmemberstothired'] = modedata['totmembers']*modedata['tothired']
        modedata['tothired2'] = modedata['tothired']**2
        datagrid[:, i, j] = scaler.transform(modedata.values.reshape(1,-1))


z = np.zeros((30,31))
for i in range(30):
    for j in range(31):
        z[i, j] = model.predict_proba([datagrid[:, i, j]])[0,1]


plt.clf()
hm = sns.heatmap(z, vmin = 0, annot=False, fmt='.0%')
hm.set_yticklabels(np.arange(1, 31, 2))
plt.ylabel('total members (without hired personnel)')
plt.xlabel('total hired personal')
plt.title('Expedition success chance by team size')
plt.show()