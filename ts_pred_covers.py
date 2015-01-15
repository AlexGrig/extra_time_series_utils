# -*- coding: utf-8 -*-
"""
Predictor based on Random Forest
"""
import numpy as np
import scipy as sp
import warnings

import data_utils as du # my normalization module
from sklearn.ensemble import RandomForestRegressor


import extra_ls_solvers as ls_solve

class Struct(object): # Needed for emulation of structures
    pass    

class RandomForest(object):
    
    def __init__(self, **kwargs):
       self.kwargs = kwargs
       self.rf_reg = RandomForestRegressor(**kwargs)
       self.data = Struct()
    
       self.model_built = False
       self.data_set = False
       
    def set_data(self,X,Y,normalize=False):
        
        if normalize:
            X_norm,x_means,x_stds = du.normalize(X,ntype=0) # zero mean unit variance    
            Y_norm,y_means,y_stds = du.normalize(Y,ntype=0) # zero mean unit variance                
                
            self.data.normalized = True   
             
            self.data.X = X_norm
            self.data.Y = Y_norm
            
            self.data.x_means = x_means
            self.data.x_stds = x_stds
            
            self.data.y_means = y_means
            self.data.y_stds = y_stds
        else:
            self.data.normalized = False
            
            self.data.X = X
            self.data.Y = Y

        self.model_built = False
        self.data_set = True        

    
    def train(self):
        orig_shape = self.data.Y.shape
        self.data.Y.shape = (du.vector_len(self.data.Y),)
        self.rf_reg.fit(self.data.X,self.data.Y)
        self.data.Y.shape = orig_shape
        
        self.model_built = True 
        
    def predict(self,X_pred,Y_known = None):
        
        if not self.model_built:
            raise ValueError("Random Forest model: Prediction is impossible model is not trained.")
        
        if self.data.normalized:            
            (X_d,tmp1,tmp2) = du.normalize( X_pred, None, self.data.x_means,self.data.x_stds )
        else:
            X_d = X_pred
        
        Y_pred = self.rf_reg.predict(X_d)
        if self.data.normalized: 
            Y_pred = du.denormalize( Y_pred,  self.data.y_means,self.data.y_stds )
                    
            
        if Y_known is None:
            return (Y_pred, None)
        else:
            Y_pred.shape = Y_known.shape # shape of prediction is (l,)
            return (Y_pred,  np.mean( np.power( Y_pred - Y_known, 2), axis=0  ) )
            
    def copy(self):
        this_class = type(self)     
        new_instance = this_class(**self.kwargs)
        
        return new_instance
        
class ts_KNN(object):
    def __init__(self, **kwargs):
       self.kwargs = kwargs
       
       if not 'order' in kwargs:
           raise ValueError( "ts_KNN: Number of nearest neighbours is not assigned." )
           
       self.order = kwargs[ "order" ]
       self.data = Struct()
    
       self.model_built = False
       self.data_set = False
       
    def set_data(self,X,Y):
                    
        if X.shape != Y.shape:
            raise ValueError( "ts_KNN: Number of nearest neighbours is not assigned." )                        
        
        self.data.X = X
        self.data.Y = Y
        self.data.dim = X.shape[1] # dimensionality of data
        self.data.samples = X.shape[0]
      
        self.model_built = False
        self.data_set = True        

    def train(self):
        """
        Dummy method but is kept here for interface compatibility
        """
        
        self.model_built = True 
        
    def predict(self,X_pred,Y_known = None):
        
        
        Y_pred = np.empty( X_pred.shape )        
        
        for j in xrange( X_pred.shape[0] ):
            neighbours_dists = []
            forec = X_pred[j,:]
            for i in xrange( self.data.samples ):
                A = np.hstack( ( np.ones( (self.data.dim,1) ), np.atleast_2d( self.data.X[i,:] ).T ) )
            
                result = ls_solve.ls_cof( A, forec, check_finite = False )            
                            
                res = np.sum( np.power( forec - np.dot( A, result[0]), 2 ) ) # squared norm of residuals
                
                neighbours_dists.append( (i,res) )
                
            nn_inds = [ i[0] for i in sorted(neighbours_dists, key=lambda x:x[1])[0:self.order] ]
            
            U = np.empty( (self.data.dim,  self.order ) ) # modelling regressors
            V = np.empty( (self.data.dim,  self.order ) ) # modelling forecasts
            
            for i in xrange(self.order):
                A = np.hstack( ( np.ones( (self.data.dim,1) ), np.atleast_2d( self.data.X[nn_inds[i],:] ).T ) )
                result = ls_solve.ls_cof( A, forec, check_finite = False )
                U[:,i] = np.dot( A, result[0] )                    
                
                B = np.hstack( ( np.ones( (self.data.dim,1) ), np.atleast_2d( self.data.Y[nn_inds[i],:] ).T ) )                  
                V[:,i] = np.dot( B, result[0] )            
            
            rank = self.order
            if (self.order > 1):
                
                (R,col_perm) = sp.linalg.qr( V, pivoting=True,mode='r' )                
                rank_reveal = ( np.abs(np.diag(R) / R[0,0]) < 1e-10 )
                if np.any(rank_reveal):
                     rank = self.order - len(np.nonzero(rank_reveal)[0])
                     
                     warnings.warn("""ts K-NN rank deficient NN combination.
                                        Reduce number of NNs from %i to %i.""" % \
                        ( self.order, rank ), RuntimeWarning)
                    
                     U = U[:,col_perm]; U = U[:,rank_reveal] # select only linearly independent conlumns
                     V = V[:,col_perm]; V = V[:,rank_reveal] # select only linearly independent conlumns
                
                UTU = np.dot(U.T, U)
                
                ones = np.ones((rank,))
                lu_dec = sp.linalg.lu_factor( UTU, check_finite=False )
                
                denom = sp.linalg.lu_solve( lu_dec, ones, check_finite=False ); denom = np.dot( ones.T, denom)
                tmp = np.dot(U.T,forec)
                numer = sp.linalg.lu_solve( lu_dec, tmp, check_finite=False); numer = np.dot( ones.T, numer) - 1
                
                weights = sp.linalg.lu_solve( lu_dec, tmp - numer/denom*ones, check_finite=False)
                
                Y_pred[j,:] = np.dot( V, weights ).T
                
            else:
                Y_pred[j,:] = V[:,0].T               
       
        if Y_known is None:
            return (Y_pred, None)
        else:
            Y_pred.shape = Y_known.shape # shape of prediction is (l,)
            return (Y_pred,  np.mean( np.power( Y_pred - Y_known, 2), axis=0  ) )
        
            
    def copy(self):
        this_class = type(self)     
        new_instance = this_class(**self.kwargs)
        
        return new_instance