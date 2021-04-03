import numpy as np


class ComplexLoss:
    ''' 
    This loss function can be used for multiclass
    The alpha is no use while num_class > 2
    Now this function only support lightgbm and classes > 2
    It will support xgboost and binary loss function later 
    '''

    def __init__(self,alpha = 0.75,gamma = 0.5,smoothing_value = 0.1,num_class = 3,tree_type = 'lgb'):
        if alpha <= 0 or alpha >=1:
            raise Exception("alpha must in (0,1) ")
        self.alpha = alpha
        if gamma < 0:
            raise Exception('gamma must > 0')
        self.gamma = gamma
        if smoothing_value < 0 or smoothing_value >= 1:
            raise Exception('smoothing_value should be in [0,1)')
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
        pred = np.reshape(y_pred, (len(target), self.num_class), order='F')
        '''label smoothing'''
        target = (1 - self.smoothing_value) * target + self.smoothing_value / self.num_class 
        eps = 1e-6
        """get softmax probability"""
        pred = np.exp(pred)
        softmax_p = np.multiply(pred, 1/np.sum(pred, axis=1)[:, np.newaxis])
        """get focal loss"""
        """The alpha is no use while num_class > 2"""
        softmax_p =  ((1 - softmax_p) ** self.gamma) * softmax_p
        assert target.any() >= 0 or target.any() <= self.num_class
        grad = (softmax_p - target)
        hess = 2.0 * softmax_p * (1.0-softmax_p) + eps
        hess[hess <= eps] = eps
        return grad.flatten('F'), hess.flatten('F')
