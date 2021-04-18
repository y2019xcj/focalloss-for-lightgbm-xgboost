# focal loss in lightgbm(xgboost) for multi-class
This loss function contains focal loss[1],now only support lightgbm for multi-class(classes > 3,it will support xgboost and binary class task later)

### focal loss
and alpha,gamma is the parameter of focal loss,which is:

![image](https://user-images.githubusercontent.com/55391817/113478054-f4fc4980-94b8-11eb-81b5-b10014af1a53.png)

alpha is used for imbalanced sample(It's no use while in multi-class task),and gamma is used for hard-to-learn sample,and in multi-class problem,it's seems that the alpha is no use.

in xgboost/lightgbm,we should provide the calculate formula of grad and hess.

while the first derivative of multi-class loss function is(From: https://zhuanlan.zhihu.com/p/149419189):
     ![image](https://user-images.githubusercontent.com/55391817/115137525-3fed9380-a059-11eb-9484-977b81dbda8c.png)
     
     
     
and the second derivative is:

![image](https://user-images.githubusercontent.com/55391817/115137577-7a573080-a059-11eb-8107-dfe706afa357.png)

   
   
To speed up the calculate of loss function,we calculate this formula by numpy.

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
### 4.get probability result in your dataset
Here val_data is your validation dataset.

```
lgb_prob = clf_lgb.predict_proba(val_data)
lgb_prob = np.exp(lgb_prob)
results_lgb_prob = np.multiply(lgb_prob, 1/np.sum(lgb_prob, axis=1)[:, np.newaxis])
```
and results_lgb_prob is the probability of your validation dataset.

Ref:

[1] Lin T Y ,  Goyal P ,  Girshick R , et al. Focal Loss for Dense Object Detection[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2017, PP(99):2999-3007.
