from typing import List, Union, Tuple, Optional, Callable
import numpy as np
from abc import ABC, abstractmethod

class Abstract_Sorting:
    """
        This class is defined to perform a Single Objective Optimization based by building blocks of 
        d*r instances of the BBOB functions, where d is the intrinsic dimensionality and 
        r is the number of repetitions. 

        Each target corresponds to the arithmetic mean of multiple instances of the BBOB defined on 
        some latent dimensions.
    """
    @abstractmethod
    def __init__(self, intrinsic_dimension:int,
                 increasing:bool):
        r"""
        This is the construction of the abstract class. This takes the following parameters:
            
            Args:
            -------------
            - intrinsic_dimension: The intrinsic dimensionality of the module (positive integer)
            - increasing: A boolean value to determine if the sorting is increasing or decreasing    
        """

        # Set the intrinsic dimensionality and repetitions
        # as properties of this class
        self.intrinsic_dimension = intrinsic_dimension
        self.increasing = increasing
        pass
    
    @abstractmethod
    def __call__(self, array_of_arrays:Union[list,np.ndarray], 
                 **kwds)->np.ndarray:
        r"""
        The `__call__` method of this class just takes an array of arrays and sorts them
        and informs the user if the array is sorted or not. This is to use the class as a callable object 
        when passing to an Evolutionary Algorithm.

        Args:
        -------------  
        - array_of_arrays: A list of arrays to be sorted
        - **kwds: Additional keyword arguments to pass to the function

        ###
        As this is a constructor, this method should return a `NumPy` array of arrays as 
        a setup for other methods
        """
        
        # Use this as some template to work with the array of arrays as a NumPy array

        if isinstance(array_of_arrays, np.ndarray):
            # Make a copy if this is true
            return_array = array_of_arrays.copy()
        else:
            return_array = np.array(array_of_arrays,
                              dtype=float,
                              copy=True,
                              ndmin=2,
                              subok=False)
        return return_array

    
    @abstractmethod
    def __sort_array(self, 
                     sub_array:Union[List[float],np.ndarray],
                     **kwargs)->np.ndarray:
        pass

    @abstractmethod
    def __is_sorted(self, 
                    sub_array:Union[List[float],np.ndarray],
                    **kwargs)->bool:
        pass

    @abstractmethod
    def is_feasible(self, 
                    sub_array:Union[List[float],np.ndarray],
                    f:float,
                    **kwargs)->bool:
        r"""
        This is a method to check if the array is feasible. This is to be used in the
        framework of CMA-ES to check if the array is feasible or not. This should be 
        passed via the CMAOptions.

        Args:
        ------------
        - sub_array: The array to be checked
        - f: The objective function result to be used
        """
        
        pass
    
    @property
    def intrinsic_dimension(self):
        return self.__intrinsic_dimension
    
    @intrinsic_dimension.setter
    def intrinsic_dimension(self,new_intrinsic_dimension:int):
        if isinstance(new_intrinsic_dimension,int) and new_intrinsic_dimension > 0:
            self.__intrinsic_dimension = new_intrinsic_dimension
        else:
            raise AttributeError("The intrinsic dimension must be a positive integer")
        
    @property
    def increasing(self):
        return self.__increasing
    
    @increasing.setter
    def increasing(self,new_increasing:bool):
        if isinstance(new_increasing,bool):
            self.__increasing = new_increasing
        else:
            raise AttributeError("The increasing property must be a boolean value")
    
        