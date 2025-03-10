import sys, os


try:
    from mylib.lib_BAxUS.BAxUS.baxus.benchmarks.benchmark_function import SyntheticBenchmark
except ModuleNotFoundError as e:
    print("The benchmark module could not be found! , trying to use another way!!!")
    __set_directory__:str= os.path.abspath(os.path.dirname(__file__))

    sys.path.append(os.path.join(__set_directory__,"mylib","lib_BAxUS","BAxUS"))
    from baxus.benchmarks.benchmark_function import SyntheticBenchmark

import ioh
from typing import List,Tuple,Optional,Union, Callable

import numpy as np



# This is a helper class to wrap the IOH-BBOB problem instances to run with the BAxUS framework
class IOH_BAxUS_Wrapper(SyntheticBenchmark):

    r"""
    This is a wrapper which extends the IOH Single Objective defined instances
    to work with BAxUS.
    """
      
    def __init__(self,
                 f:Callable,
                 dim:int,
                 lb:int,
                 ub:int,
                 noise_std: Optional[float] = None,
                 )->None:

        r"""
        Args:
        --------
        - f: A IOH Real Single Objective instance
        - noise_std: Standard deviation of the observation noise.
        - negate: If True, negate the function.
        """
        self.__dim = dim

        self.__bounds = [(lb, ub) for _ in range(self.__dim)]

        self.__f:ioh.problem.RealSingleObjective = f

        # Convert the bounds definition
        lb_:np.ndarray = np.ravel(np.array(self.__bounds)[:,0])
        ub_:np.ndarray = np.ravel(np.array(self.__bounds)[:,1])

        # Use the constructor from `SyntheticBenchmark`
        super().__init__(dim= self.__dim,lb=lb_,ub=ub_,
                         noise_std=noise_std)
    
    def __call__(self, x: Union[np.ndarray, List[float], List[List[float]]])->np.ndarray: 

        r"""
        This is the implementation of the `__call__` function for this wrapper class.

        Args:
        ---------
        - x: A list or array with possible points to evaluate.
        """
        # Call the __call__ function from the polymorphic class


        #x =super().__call__(x)
        x:np.ndarray = np.array(x)
        if x.ndim == 0:
            x = np.expand_dims(x, 0)
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        assert x.ndim == 2

        result = self.__f(x.ravel())
 
        return result
        

    @property
    def dim(self)->int:
        return self.__dim
    
    @property
    def bounds(self)->Union[float,List[float]]:
        return self.__bounds
    
    @property
    def IOH_instance(self)->ioh.problem.RealSingleObjective:
        return self.__f



