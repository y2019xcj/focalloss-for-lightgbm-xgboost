# loss-function-for-lightgbm-xgboost
now only support lightgbm for multi-class(classes > 3)
## usage:

### 1.import loss function lib
```
import lightgbm as lgb
import lossfunction as lf
import numpy as np
```
### 2.init loss function
```
focal_loss_lgb = lf.ComplexLoss(gamma = 0.5)
param_dist= {'objective':focal_loss_lgb.focal_loss}
param_dist['num_class'] = '3'
clf_lgb = lgb.LGBMClassifier(**param_dist,random_state=2021)
```
### 3.train your dataset
```
clf_lgb.fit(X_train, y_train)
```
### 4.get probability result
```
results_lgb_prob = clf_lgb.predict_proba(val_data.iloc[:,:factor_nums])
results_lgb_prob = np.exp(results_lgb_prob)
results_lgb_prob = np.multiply(results_lgb_prob, 1/np.sum(results_lgb_prob, axis=1)[:, np.newaxis])
```
