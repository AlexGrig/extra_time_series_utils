# -*- coding: utf-8 -*-
"""
Time series forecasting class.
"""

import numpy as np

# nonstandard imports:
from utils import data_utils as du

class TimeSeries(object):
    """
    Class implements useful functions for time series manipulations.
    
    TODO: implement iterator interface
    """
    
    def __init__(self, train_data, test_data = None, normalize = False):
        """
        Constructor.
        
        Input:
            train_data - train data of the time series. 
            test_data - test data for the time series. Needed for
            
            normalize - boolean, whether time series data is normalized to zero
                        mean, unit variance.            
        """        
    
        if train_data is None: 
            raise ValueError("TimeSeries class: train_data can't be empty")
          
        train_len = self.validate_ts_data( train_data )  
        
        self._train_data = train_data.copy() # Copy train data, original var is not modified
        self._train_data.shape = (train_len,) # Reshsape train data
        self._train_len = train_len
     
        if  test_data is None:
            test_data = None
            self._test_len = 0
        else:
            test_len = self.validate_ts_data( test_data ) 
            
            self._test_data = test_data.copy() # Copy train data, original var is not modified
            self._test_data.shape = (test_len,) # Rehsape train data
            self._test_len = test_len
            
        if normalize:
            (result, means, stds) = du.normalize( self._train_data, ntype = 0) # normalize with zero mean unit variance
            self._train_data = result
            self._means = means
            self._stds = stds
            
            if test_data is not None:
                (result, means, stds) = du.normalize( self._test_data, means = means, stds = stds)
                self._test_data = result
            del result, means, stds
        else:
            self._means = 0.0
            self._stds = 1.0
        
    def validate_ts_data( self, data ):
        """        
        Function checks that the data is one dimensional and return its length.
        Exception is raised if the data is not one dimensional.
        
        This is a substitution for np.squeeze() function which doesn't modify original
        data. It could work improperly with array of shape > 2.
        
        Input:
            data - numpy array
            
        Output:
            length - length of the one dimensional data            
        """
        
        #data1 = np.asarray_chkfinite(data) # Check for NaNs and Infs and convert to ndarray
        data1 = data
        
        shape = data1.shape
        
        if len(shape) > 2:
            raise ValueError("TimeSeries class: train_data can't be more than 2 dimensional")
            
        length = du.vector_len(data)
        
        return length
                       
        
    def get_train(self):
        """
        Function which return the train data. Used in the property.
        """
    
        return self._train_data
    
    
    def get_test(self):
        """
        Function which return the test data. Used in the property.
        """
    
        return self._test_data
        
    def get_train_len(self):
        """
        Function which return the train data length. Used in the property.
        """
    
        return self._train_len
    
    def get_test_len(self):
        """
        Function which return the test data length. Used in the property.
        """
    
        return self._test_len
    
    train_data = property(get_train) # for other operations like set exception is raised
    test_data = property(get_test)
    train_len = property(get_train_len)
    test_len = property(get_test_len)
    
    def denormalize(self, data=None):    
        """
        Denormalize training and test data.
        Modify data in time series and return denormalized training data
        
        """
    
        # TODO: add variablle _normalized and modify stds, means accordingly
    
        if data is None:
            data = self._train_data
            
        result = du.denormalize( data, self._means, self._stds)        
    
        if data is None:
            self._train_data = result
            self._means = 0.0
            self._stds = 1.0
        
        return result
    
    def ts_matrix(self, regressor_size, prediction_horizon, only_b = False, num_b_col = None, with_nans = False, data = None ):
        """
         In the model Ax = B return matrices A and B. Each row of B
         are elments (1,2...prediction_horizon) points ahead of 
         corresponding predictors in the rows of A.
                
         Number of columns in B equals prediction_horizon. 
            
         Input:
             regressor_size - number of regresssors (predictors). Equals
                              number of columns in the matrix A.
             prediction_horizon - how many time steps ahead we want to
                                   build a model for. Must be >= 1.
             only_b - whether we need only B matrix
             num_b_col - only if last column of B needed
             with_nans - whether we include nans into the  matrix B for longer prediction
                         horizons.
             data - data for which matrices are build. If empty
                    training data is taken.
    
        """
        
        if not num_b_col is None:
            if num_b_col > prediction_horizon:
                raise ValueError("TimeSeries.ts_matrix : num_b_col can not be greater than prediction_horizon" )
        
        if data is None:
            data = self._train_data
            length = self._train_len
        
        else:
            length = self.validate_ts_data( data )
            
            data_old_shape = data.shape
            data.shape = (length,)
        
        if with_nans:
            data = np.concatenate( (data, np.nan* np.empty( (prediction_horizon, ) ) ) )
            length = length + (prediction_horizon)
           
        m = regressor_size + prediction_horizon;               
        n = length - prediction_horizon - regressor_size + 1 
        if n < 1:
            raise ValueError("TimeSeries.ts_matrix: pridiction_horizon %i is too large" % prediction_horizon)
      
        A = None; B = None 
        if not only_b:
            
            if num_b_col is None:
                j_val = range(0,m)
                TM = np.zeros( (n,m) ) # Matrix preallocation
            else:
                j_val = range(0,regressor_size) + range( (m - num_b_col) ,m)
                TM = np.zeros( (n,regressor_size+num_b_col) ) # Matrix preallocation
                
            for (ind,j) in enumerate(j_val):
               TM[:,ind] = data[ j:(j+n)]                  

            A = TM[:,0:regressor_size ]
            B = TM[:, regressor_size:]
               
        else: # return only B
            if num_b_col is None:
                TM = np.zeros( (n,prediction_horizon )) # Matrix preallocation
                
                for (ind,j) in enumerate( xrange(regressor_size,m) ):
                   TM[:,ind] = data[ j:(j+n)]                  
                
                B = TM 
            else:
                 TM = np.zeros( (n,num_b_col )) # Matrix preallocation
                
                 for (ind,j) in enumerate( xrange((m-num_b_col),m) ):
                    TM[:,ind] = data[ j:(j+n)]                  
                
                 B = TM 


        
        if locals().get('data_old_shape'):
            data.shape = data_old_shape # restore initial shape of data
            
        if A is None:
            return B
        else:
            return A,B
    
    def windows( self, window_size, data = None ):
        """
        Function return time series cut on windows.
        """
        
        if data is None:
            data = self._train_data
            
        A,B = self.ts_matrix( (window_size-1), 1, data = data )
            
        return np.hstack((A,B))
    
    
    def calc_error(true_val, predictions):
        """
        Class method. Computes error.
        
        Input:
            true_val : true values
            predictions : predictions
            
        Output:
            error : error
        """

        # MSE error
        error = np.mean( np.power( (true_val - predictions), 2) )

        return error
        
    
class RecursivePredictor(object):
            
    def __init__(self, time_series, model, regressor_size, prediction_dim, prediction_steps_no, prediction_variables=None):    
        
        self.model = model
        
        self.regressor_size = regressor_size
        self.prediction_dim = prediction_dim
        self.prediction_steps_no = prediction_steps_no
        self.time_series = time_series
        
        # data for which predictions are made, predictions are made for each row
        if prediction_variables is None:
            self.orig_pred_variables,self.orig_pred_targets = self.time_series.ts_matrix(self.regressor_size,self.prediction_dim*self.prediction_steps_no,\
                                        only_b=False, num_b_col=None, with_nans = False, data= self.time_series.test_data ) 
        else:
            self.orig_pred_variables = prediction_variables
            self.orig_pred_targets = None
            
        self.iteration_no = 0 # number of iteration        
        self.prediction_variables = self.orig_pred_variables.copy()
        
        self.pred_from_prev_step = None # predictions from previous step
    
    def _get_orig_pred_data(self):
        """
        Returns the values of self.orig_pred_variables, self.orig_pred_targets
        variables. These variables were originally set as prediction variables.
        Other prediction variables: self.prediction_variables change with time.
        """
        return self.orig_pred_variables,self.orig_pred_targets    
    orig_pred_data = property(_get_orig_pred_data) # the same function as a property.
    
    def _get_prediction_vars(self, pred_vars):
        """
        Function return variables for which predictions are made. This function
        is needed because time series variables maybe be changed for prediction
        in child classes.
        
        """
        
        return pred_vars
        
    def predict_next(self, update_prediction_vars = None):
        """
        
        Input:
            update_prediction_vars - 
        """
        
        self.iteration_no += 1
        
        if self.iteration_no == 1: # first iteration 
            
            X_train,Y_train = self._get_train_data()
                                        
            self.model.set_data( X_train,Y_train )
            if hasattr(self.model,'set_ts_pred_iteration'):  # for TS models which do regression iterations
                self.model.set_ts_pred_iteration( self.iteration_no )
                
            self.model.train()
            #self.model.is_stable() # TODO: remove this line
            if not self.orig_pred_targets is None:
                Y_predict, MSE = self.model.predict(self._get_prediction_vars( self.prediction_variables ),self.orig_pred_targets[:,0:self.prediction_dim])
            else:
                Y_predict,MSE = self.model.predict( self._get_prediction_vars( self.prediction_variables ) )                
        else:
            if not update_prediction_vars is None:
                 if update_prediction_vars.shape[1] != self.prediction_dim:
                     raise ValueError("Recursive predictor: dimensions of update_prediction_vars do not coincide.")
                 self.prediction_variables = np.hstack(    (self.prediction_variables[:,self.prediction_dim:],update_prediction_vars )   )
            else:
                 self.prediction_variables = np.hstack(    (self.prediction_variables[:,self.prediction_dim:],self.pred_from_prev_step )   )               
                
            if not self.orig_pred_targets is None:
                start_ind = (self.iteration_no-1)*self.prediction_dim
                end_ind = self.iteration_no*self.prediction_dim
                (Y_predict, MSE) = self.model.predict(self._get_prediction_vars( self.prediction_variables ), \
                                                    self.orig_pred_targets[:,start_ind:end_ind] )
            else:
                Y_predict,MSE = self.model.predict(self._get_prediction_vars( self.prediction_variables ))
                
        self.pred_from_prev_step = Y_predict.copy()
        
        return Y_predict, self.prediction_variables.copy(), MSE
        
    def _get_train_data(self):
        X_train,Y_train = self.time_series.ts_matrix(self.regressor_size,self.prediction_dim,\
                                        only_b=False, num_b_col=None, with_nans = False)
        return X_train,Y_train
        
class DirectPredictor(RecursivePredictor):
    
    def _get_train_data(self):
        
        iteration_no = self.iteration_no
        X_train,Y_train = self.time_series.ts_matrix(self.regressor_size,self.prediction_dim*iteration_no,\
                                        only_b=False, num_b_col=self.prediction_dim, with_nans = False)
        return X_train,Y_train
    
    def predict_next(self):

        self.iteration_no += 1
        
        X_train,Y_train = self._get_train_data()
        
        model = self.model.copy()
        model.set_data( X_train,Y_train )
        if hasattr(model,'set_ts_pred_iteration'):  # for TS models which do regression iterations
            model.set_ts_pred_iteration( self.iteration_no )
        model.train()
        
        if not self.orig_pred_targets is None:
            start_ind = (self.iteration_no-1)*self.prediction_dim
            end_ind = self.iteration_no*self.prediction_dim
            (Y_predict, MSE) = model.predict(self._get_prediction_vars( self.prediction_variables ), \
                                                self.orig_pred_targets[:,start_ind:end_ind] )
        else:
            Y_predict,MSE = model.predict(self._get_prediction_vars( self.prediction_variables ))
                

        return Y_predict, self.prediction_variables.copy(), MSE

        
class DiRRecPredictor(RecursivePredictor):
    """
    
    """

    def _get_train_data(self):
        #TODO: pass parameters such as  self.regressor_size, self.prediction_dim ...
        # as parameters of the function. It is easier to rewrite this  method in the child class.
        iteration_no = self.iteration_no
        X_train,Y_train = self.time_series.ts_matrix(self.regressor_size + (iteration_no-1)*self.prediction_dim,self.prediction_dim,\
                                        only_b=False, num_b_col=None, with_nans = False)
        return X_train,Y_train
     
    def predict_next(self,update_prediction_vars = None):

        self.iteration_no += 1
        
        X_train,Y_train = self._get_train_data()
        
        model = self.model.copy()
        model.set_data( X_train,Y_train )
        if hasattr(model,'set_ts_pred_iteration'): # for TS models which do regression iterations
            model.set_ts_pred_iteration( self.iteration_no )
        model.train()
                
        if (self.iteration_no > 1):
            if (not update_prediction_vars is None):
                 if update_prediction_vars.shape[1] != self.prediction_dim:
                     raise ValueError("Recursive predictor: dimensions of update_prediction_vars do not coincide.")
                 self.prediction_variables = np.hstack(    (self.prediction_variables,update_prediction_vars )   )
            else:
                 self.prediction_variables = np.hstack(    (self.prediction_variables,self.pred_from_prev_step )   )               
        
        
        if not self.orig_pred_targets is None:
            start_ind = (self.iteration_no-1)*self.prediction_dim
            end_ind = self.iteration_no*self.prediction_dim
            (Y_predict, MSE) = model.predict(self._get_prediction_vars( self.prediction_variables ), \
                                                self.orig_pred_targets[:,start_ind:end_ind] )
        else:
            Y_predict,MSE = self.model.predict(self._get_prediction_vars( self.prediction_variables ))
                
        self.pred_from_prev_step = Y_predict.copy()
        
        return Y_predict, self.prediction_variables.copy(), MSE

        
if __name__ == "__main__":
    
    dd = np.arange(0,10)
    dd.shape = (1,10)    
    
    ts = TimeSeries( dd, normal=False)
    
    #ts.ts_matrix(3,7)
    ts.ts_matrix_true(3,2, with_nans = True)
    
    
    