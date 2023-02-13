
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

filename1 = "background.csv"
filename2 = "signal.csv"
df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)
df1 = df1.drop(columns=['Unnamed: 0'],axis=1)
df2 = df2.drop(columns=['Unnamed: 0'],axis=1)
#scaler = Normalizer()

#for c in df1.columns:
#    df1[c] = (df1[c] - df1[c].mean())/df1[c].std()
#    df2[c] = (df2[c] - df2[c].mean())/df2[c].std()

df1['label'] = np.zeros(df1.shape[0])
df2['label'] = np.ones(df2.shape[0])

new_df = [df1,df2]

df = pd.concat(new_df)

y  = df[['label']]
X  = df.drop(columns=['label'],axis=1)

X=X.to_numpy()
y=y.to_numpy()


X_train, X_valid, y_train, y_valid =  train_test_split(X,y,random_state=1,test_size=.7)
#X_valid, X_test, y_valid, y_test =  train_test_split(X_valid,y_valid,random_state=1,test_size=.7)


scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
#X_test = scaler.transform(X_test)


rand = xgb.XGBClassifier(max_depth=8,sampling_method='uniform',eta=0.15,alpha=.4,eval_metric='rmse',n_jobs=8,tree_method='approx')#criterion='gini',min_samples_leaf=5,max_depth=6,n_jobs=-1)
rand.fit(X_train,y_train)
y_pred_xgb = rand.predict_proba(X_valid)


plt.hist(y_pred_xgb[:,0],label='background',bins=50,histtype='step')
plt.hist(y_pred_xgb[:,1],label='signal',bins=50,histtype='step')
plt.legend()
plt.xlabel('BDT Output')
plt.ylabel('Counts`')
plt.yscale('log')
#plt.gca().set_aspect('equal', adjustable='box')
plt.show()


fpr_xgb, tpr_xgb, thresholds = roc_curve(y_valid.ravel(), y_pred_xgb[:,1].ravel())
auc_xgb = auc(fpr_xgb, tpr_xgb)
plt.plot(tpr_xgb, 1/(fpr_xgb+1e-5),label=f'XGB, AUC={auc_xgb:.2f}')
#plt.plot(tpr_xgb, 1/(fpr_xgb+.000001),label=f'RandomForestClassifier, AUC={auc_xgb:.2f}')
plt.yscale('log')
plt.xlabel('Signal Efficiency')
plt.ylabel('Background Rejection')
plt.xlim([0.01, 1.0])
#plt.gca().set_aspect('equal', adjustable='box')
plt.legend()#loc='lower left',title_fontsize='x-small')
plt.show()
