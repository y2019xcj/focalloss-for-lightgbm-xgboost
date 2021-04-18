import numpy as np


class ComplexLoss:
    ''' 
    This loss function can be used for multiclass
    The alpha is no use while num_class > 2
    Now this function only support lightgbm and classes > 2
    It will support xgboost and binary loss function later 
    '''

    def __init__(self,gamma = 0.5,num_class = 3,tree_type = 'lgb'):
        if alpha <= 0 or alpha >=1:
            raise Exception("alpha must in (0,1) ")
        self.alpha = alpha
        if gamma < 0:
            raise Exception('gamma must > 0')
        self.gamma = gamma
        self.smoothing_value = smoothing_value
        if tree_type != 'lgb':
            raise Exception('only support lightgbm now')
        self.tree_type = tree_type
        if num_class <= 2:
            raise Exception('num_class must > 2')
        self.num_class = num_class

    def focal_loss(self,y_true,y_pred):

 
        grad = np.zeros((len(y_true), self.num_class), dtype=float)
        hess = np.zeros((len(y_true), self.num_class), dtype=float)

        target = np.eye(self.num_class)[y_true.astype('int')]
        pred = np.reshape(y_pred, (len(y_true), self.num_class), order='F')
        # """get softmax probability"""
        softmax_p = np.exp(pred)
        softmax_p = np.multiply(softmax_p, 1/np.sum(softmax_p, axis=1)[:, np.newaxis])
        for c in range(pred.shape[1]):
            pc = softmax_p[:,c]
            pt = softmax_p[:][target == 1]
            grad[:,c][target[:,c] == 1] = (self.gamma * np.power(1-pt[target[:,c] == 1],self.gamma-1) * pt[target[:,c] == 1] * np.log(pt[target[:,c] == 1]) - np.power(1-pt[target[:,c] == 1],self.gamma) ) * (1 - pc[target[:,c] == 1])
            grad[:,c][target[:,c] == 0] = (self.gamma * np.power(1-pt[target[:,c] == 0],self.gamma-1) * pt[target[:,c] == 0] * np.log(pt[target[:,c] == 0]) - np.power(1-pt[target[:,c] == 0],self.gamma) ) * (0 - pc[target[:,c] == 0])
            hess[:,c][target[:,c] == 1] = (-4*(1-pt[target[:,c] == 1])*pt[target[:,c] == 1]*np.log(pt[target[:,c] == 1])+np.power(1-pt[target[:,c] == 1],2)*(2*np.log(pt[target[:,c] == 1])+5))*pt[target[:,c] == 1]*(1-pt[target[:,c] == 1])
            hess[:,c][target[:,c] == 0] = pt[target[:,c] == 0]*np.power(pc[target[:,c] == 0],2)*(-2*pt[target[:,c] == 0]*np.log(pt[target[:,c] == 0])+2*(1-pt[target[:,c] == 0])*np.log(pt[target[:,c] == 0]) + 4*(1-pt[target[:,c] == 0])) - pc[target[:,c] == 0]*(1-pc[target[:,c] == 0])*(1-pt[target[:,c] == 0])*(2*pt[target[:,c] == 0]*np.log(pt[target[:,c] == 0]) - (1-pt[target[:,c] == 0]))

        return grad.flatten('F'), hess.flatten('F')
