from typing import List, Union, Tuple, Optional
from .Abstract_Sorting import Abstract_Sorting
import numpy as np

class One_Dimensional_Intrinsic_Sorting(Abstract_Sorting):
    r"""
        This class is defined to perform a Single Objective Optimization based by building blocks of 
        1D instances of the BBOB functions. 

        Each target corresponds to the arithmetic mean of multiple instances of the BBOB defined on 
        some latent dimensions.
    """
    def __init__(self, increasing:bool=True):
        r"""
        This is just an overload of the constructor of the `Abstract_Sorting` class.
        This just sets the intrinsic dimensionality to 1.
        """
        
        # Call the super class initializer
        super().__init__(intrinsic_dimension=1, increasing=increasing)
    
    def __call__(self, array_of_arrays:Union[list,np.ndarray], 
                 **kwds)->Union[np.ndarray,List[bool]]:
        
        # Call the super class initializer (outputs a two dimensional `NumPy`array)
        array_of_arrays_mod:np.ndarray = super().__call__(array_of_arrays, **kwds)

        # Initialize a list of boolean values and a list of arrays with the sorted arrays
        is_sorted_list:List[bool] = []
        sorted_arrays:List[Union[List[float],np.ndarray]] = []

        # Now perform the loop
        for _, sub_array in enumerate(array_of_arrays_mod):
            # Sort the array and append it to the list
            sorted_arrays.append(self.__sort_array(sub_array))

            # Check if the array is sorted and append it to the list
            is_sorted:bool = self.__is_sorted(sub_array)
            is_sorted_list.append(is_sorted)
        
        return sorted_arrays, is_sorted_list
    

    def __sort_array(self, 
                     sub_array:Union[List[float],np.ndarray],
                     **kwargs):
        
        sorted_array:np.ndarray= np.sort(sub_array, axis=None, kind='quicksort', order=None)

        if self.increasing:
            return sorted_array
        else:
            return sorted_array[::-1]
    
    def __is_sorted(self,
                    sub_array:Union[List[float],np.ndarray],
                     **kwargs)->bool:
        
        if self.increasing:
            return np.all(np.array(
                [sub_array[i] <= sub_array[i + 1] for i in range(len(sub_array) - 1)]
                ,dtype=bool))
        else:
            return np.all(np.array(
                [sub_array[i] >= sub_array[i + 1] for i in range(len(sub_array) - 1)]
                ,dtype=bool))
    
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
        
        # Check the result of the objective function first. This is a replication
        # from CMA-ES (Hansen's) specifications
        if f in (np.inf, np.nan,None):
            return False
        else:
            # Check if the array is feasible by checking it is sorted
            return self.__is_sorted(sub_array,**kwargs)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        # Add any non-picklable attributes that need to be excluded here
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)