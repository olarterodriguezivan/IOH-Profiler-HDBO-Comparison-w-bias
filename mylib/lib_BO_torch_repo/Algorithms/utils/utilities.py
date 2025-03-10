r"""
This is a utility module that contains utility functions that are used in the algorithms module.
"""

import numpy as np
import os
from typing import List, Tuple, Optional, Union, Callable


def generate_random_sample(lower_bound:float, upper_bound:float) -> float:
    """
    Generate a random sample of size n from a uniform distribution between lower_bound and upper_bound.

    Args:
    --------
    lower_bound: `float`: The lower bound of the distribution
    upper_bound: `float`: The upper bound of the distribution

    Returns:
    --------
    `np.ndarray`: A numpy array of size n with samples from the uniform distribution.
    """
    return np.random.uniform(low=lower_bound, high=upper_bound, size=1)

def generate_random_array(lower_bounds:Union[float,List[float],np.ndarray], 
                          upper_bounds:Union[float,List[float],np.ndarray], 
                          n:int,
                          shape:Optional[Tuple]) -> np.ndarray:
    """
    Generate a random sample of size n from a uniform distribution between lower_bound and upper_bound.

    Args:
    --------
    lower_bound: `float`: The lower bound of the distribution
    upper_bound: `float`: The upper bound of the distribution
    n: `int`: The number of samples to generate.
    shape: `Optional[Tuple]`: The shape of the array to be generated.

    Returns:
    --------
    `np.ndarray`: A numpy array of size n with samples from the uniform distribution.
    """
    # Check if lower_bounds is a scalar
    if isinstance(lower_bounds, (float, int)):
        lower_bounds = [lower_bounds]*n
    
    # Check if upper_bounds is a scalar
    if isinstance(upper_bounds, (float, int)):
        upper_bounds = [upper_bounds]*n
    
    
    return np.random.uniform(low=lower_bounds, high=upper_bounds, size=n)

def hill_valley_test(x_0:Union[np.ndarray,List[float]],
                     fit_x_0:float,
                     x_1:Union[np.ndarray,List[float]],
                     fit_x_1:float,
                     func:Callable,
                     nt:Optional[int]=10,
                     tolerance:Optional[Union[np.float64,float]]=np.finfo(float).eps)->bool:
    r"""
    This is a function which performs the Hill Valley Test to
    detect if two points correspond to the same Basin of attraction.

    Args:
    --------------
    - x_0: `Union[np.ndarray,List[float]]` : The initial point to trace the line
    - fit_x_0: `float`: The function evaluation at `x_0`
    - x_1: `Union[np.ndarray,List[float]]`: The endpoint to trace the line
    - fit_x_1: `float`: The function evaluation at `x_1`
    - func: `Callable`: A callable object, with an implemented `__call__(:np.ndarray)->float`, denoting a function evaluation.
    - nt: `Optional[int]`: A number of between points to evaluate.
    - tolerance: `Optional[Union[np.float64,float]]`: A tolerance to use for the comparison (just to determine the points are very similar).

    Returns:
    --------------
    - `bool`: A boolean value, True if the points are in the same basin of attraction, False otherwise.
    """

    # Perform a check-up for the inputs
    if isinstance(x_0,list):
        # Convert
        x_0:np.ndarray = np.array(x_0,
                       dtype=float,
                       copy=True,
                       subok=False).ravel()
    
    if isinstance(x_1,list):
        # Convert
        x_1:np.ndarray = np.array(x_1,
                       dtype=float,
                       copy=True,
                       subok=False).ravel()


    # Compute a difference vector
    diff_vect:np.ndarray = np.subtract(x_1,x_0,dtype=np.float64)
    # Compute a range of points
    r_points = [*range(1,nt+1,1)]

    # This is to check the special case both points are very similar.
    if np.linalg.norm(diff_vect) <= tolerance and np.abs(fit_x_0 - fit_x_1) <= tolerance:
        return True

    # A placeholder to return the heuristic
    ret_var:bool = True

    for k in r_points:
        proportion:float = k/(nt+1.0)

        # Compute x_test
        x_test:np.ndarray = np.add(x_0,proportion*diff_vect,dtype=np.float64)

        # Compute the function evaluation at x_test
        f_test:float = func(x_test)

        if max(fit_x_0,fit_x_1) <= f_test:
            ret_var:bool = False
            break

    return ret_var

def hill_valley_test_2(x_0:Union[np.ndarray,List[float]],
                     fit_x_0:float,
                     x_1:Union[np.ndarray,List[float]],
                     fit_x_1:float,
                     func:Callable,
                     nt:Optional[int]=10,
                     tolerance:Optional[Union[np.float64,float]]=np.finfo(float).eps)->Tuple[bool,int]:
    r"""
    This is the second variant of the function which performs the Hill Valley Test to
    detect if two points correspond to the same Basin of attraction. In this case,
    instead of just returning if the hill valley test is passed or not, it also returns
    the number of function evaluations.

    Args:
    --------------
    - x_0: `Union[np.ndarray,List[float]]` : The initial point to trace the line
    - fit_x_0: `float`: The function evaluation at `x_0`
    - x_1: `Union[np.ndarray,List[float]]`: The endpoint to trace the line
    - fit_x_1: `float`: The function evaluation at `x_1`
    - func: `Callable`: A callable object, with an implementend `__call__(:np.ndarray)->float`, denoting a function evaluation.
    - nt: `Optional[int]`: A number of between points to evaluate.
    - tolerance: `Optional[Union[np.float64,float]]`: A tolerance to use for the comparison (just to determine the points are very similar).

    Returns:
    --------------
    - `Tuple[bool,int]`: A tuple containing a boolean value and the number of function evaluations.
    """

    # Perform a check-up for the inputs
    if isinstance(x_0,list):
        # Convert
        x_0:np.ndarray = np.array(x_0,
                       dtype=float,
                       copy=True,
                       subok=False).ravel()
    
    if isinstance(x_1,list):
        # Convert
        x_1:np.ndarray = np.array(x_1,
                       dtype=float,
                       copy=True,
                       subok=False).ravel()


    # Compute a difference vector
    diff_vect:np.ndarray = np.subtract(x_1,x_0,dtype=np.float64)
    # Compute a range of points
    r_points = [*range(1,nt+1,1)]

    # This is to check the special case both points are very similar.
    if np.linalg.norm(diff_vect) <= tolerance and np.abs(fit_x_0 - fit_x_1) <= tolerance:
        return True, 0

    # A placeholder to return the heuristic
    ret_var:bool = True

    for k in r_points:
        proportion:float = k/(nt+1.0)

        # Compute x_test
        x_test:np.ndarray = np.add(x_0,proportion*diff_vect,dtype=np.float64)

        # Compute the function evaluation at x_test
        f_test:float = func(x_test)

        if max(fit_x_0,fit_x_1) <= f_test:
            ret_var:bool = False
            break

    return ret_var, k
