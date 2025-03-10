r"""
utf-8

This is a handle to deal with the algorithm class to implement with Modular Problems 
"""

__author__ = ["Iván Olarte Rodríguez"]

from typing import Union, Callable, List, Tuple, Optional
from abc import ABC, abstractmethod
from ioh.iohcpp.problem import RealSingleObjective, BBOB
from ioh.iohcpp import MAX
from ioh.iohcpp import RealBounds
from math import inf
import numpy as np
import warnings

class AbstractAlgorithm(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        r"""
        This is the constructor for any optimisation algorithm used within this framework.
        In this framework, the initialiser might receive two keywords related to the 
        maximisation/minimisation beahviour and the verbosity.
        """

        # Initialize the number of function evaluations
        self.__number_of_function_evaluations:int = 0
        
        
        # Set initial default variables
        verbose_init = kwargs.pop("verbose",False)
        maximisation_init = kwargs.pop("maximisation",False)

        # Initialise the bounds as Nonetype
        self.__bounds:np.ndarray = np.empty(shape=(1,2))

            
        # Assign the variables as members of the class
        self.verbose = verbose_init
        self.__maximisation = maximisation_init
        self.__current_best_index:int = 0

        # Initialize a storage variable for the best sample
        if self.__maximisation == True:
            self.__current_best:float = -inf 
        else:
            self.__current_best:float = inf 
        pass

        self.__problem = None
        self.__dimension = None
    
    @abstractmethod
    def __call__(self,
                 problem:Union[RealSingleObjective,BBOB, Callable],
                 dim:Optional[int], 
                 bounds:Optional[np.ndarray],
                 **kwargs):
        """
        This is a default function indicating the structure of the `_call_` method in the context of an algorithm
        """
        if isinstance(problem,BBOB) or isinstance(problem,RealSingleObjective) or issubclass(type(problem),RealSingleObjective) :
            
            # Assign the problem
            self.__problem = problem
            # Get the dimensionality from this
            self.dimension = problem.meta_data.n_variables
            self.maximisation = (problem.meta_data.optimization_type==MAX) # This is a placeholder to modify

            # Pass the bounds from the IOH definition (the setter function will adapt the input)
            self.bounds = problem.bounds
        elif isinstance(problem,Callable):
            # Assign the problem
            self.__problem = problem

            # Set the dimension to be given by the parameters
            self.dimension = dim
            
            # In case the a new maximisation default is given as a parameter
            self.maximisation = kwargs.pop("maximisation",False)

            analyzed_bounds = bounds
            if isinstance(analyzed_bounds,(np.ndarray, List[float], Tuple[float])):
                # Assign in this case
                self.bounds = analyzed_bounds
            else:
                raise AttributeError("The bounds for a callable problem were not found", name="bounds")
        
        else:
            raise AttributeError("The problem input is not well defined",
                                 name="problem",
                                 obj=problem)

        pass
    
    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return super().__repr__()
    
    def __name__(self)->str:
        " This is just to identify the name of the algorithm, which is included in the name"
        return self.__class__.__name__
    
    @abstractmethod
    def reset(self):
        # Set the number of function evaluations to 0
        self.number_of_function_evaluations = 0

        if self.maximisation:
            self.current_best = -inf
        else:
            self.current_best = inf
        
        self.__current_best_index = 0

        pass



    @property 
    def number_of_function_evaluations(self)->int:
        "This property handles the number of function evaluations"
        return self.__number_of_function_evaluations
    
    @number_of_function_evaluations.setter 
    def number_of_function_evaluations(self, new_eval:int)->None:
        "This is the setter for the number of function evaluations"
        if isinstance(new_eval,int) and new_eval >= 0:
            self.__number_of_function_evaluations = new_eval
        else:
            raise ValueError("The number of function evaluations must be a positive integer")
     
     
    
    @property
    def verbose(self)->bool:
        "Definition of verbosity of the algorithm"
        return self.__verbose
    
    @verbose.setter
    def verbose(self, new_verbosity:bool)->None:
        "Rewrite the verbosity parameter"
        self.__verbose = bool(new_verbosity)

    @property 
    def dimension(self)->Union[int,None]:
        return self.__dimension

    @dimension.deleter
    def dimension(self)->None:
        del self.__dimension

    @dimension.setter
    def dimension(self, new_dimension:Union[int,None])->None:
        if isinstance(new_dimension,int) and new_dimension > 0:
            self.__dimension = new_dimension
        elif isinstance(new_dimension,None):
            self.__dimension = new_dimension
        else:
            raise ValueError("The new dimension is oddly set")
    
    @property
    def current_best(self)->float:
        return self.__current_best
    
    @current_best.setter
    def current_best(self, new_current_best:float):
        if self.__maximisation:
            if new_current_best >= self.__current_best:
                # Assign this new value
                self.__current_best = new_current_best
            else:
                raise ValueError("The assignment is incorrect")
        else:
            if new_current_best <= self.__current_best:
                # Assign this new value
                self.__current_best = new_current_best
            else:
                raise ValueError("The assignment is incorrect")
    
    @property 
    def current_best_index(self)->int:
        return self.__current_best_index
    
    @current_best_index.setter
    def current_best_index(self, new_current_best_index:int)->None:
        if isinstance(new_current_best_index, int) and new_current_best_index >= self.__current_best_index:
            # Assign in this case
            self.__current_best_index = new_current_best_index
        else:
            raise ValueError("Something is wrong with this assignment")
    
    @property 
    def maximisation(self)->bool:
        return self.__maximisation
    
    @maximisation.setter
    def maximisation(self, new_definition:bool)->None:
        # Assign the change in variable
        if self.__maximisation != bool(new_definition):
            self.__maximisation = bool(new_definition)
            # Change given this condition
            if self.__maximisation == True:
                self.__current_best = -inf
            else:
                self.__current_best = inf

    @property
    def bounds(self)->np.ndarray:
        r"""
        Bounds property `getter`. This just a repeater from the token in memory
        """
        return self.__bounds
    
    @bounds.setter
    def bounds(self, new_bounds:Union[np.ndarray,RealBounds,List[float], Tuple[float]]):
        r"""
        This is the bounds property setter
        """

        # The first case is if the bounds come from an IOH defined problem
        if isinstance(new_bounds,RealBounds):

            lower_bounds:Union[np.ndarray,float] = new_bounds.lb
            upper_bounds:Union[np.ndarray,float] = new_bounds.ub
            
            if isinstance(lower_bounds,float):
                # this is in case the value is repeated per every dimension
                lower_bounds_arr = np.ones(shape=(self.dimension,1))*lower_bounds
                upper_bounds_arr = np.ones(shape=(self.dimension,1))*upper_bounds

                # Reshape the array
                self.__bounds:np.ndarray = np.hstack(tup=(lower_bounds_arr,upper_bounds_arr))
            
            elif isinstance(lower_bounds, np.ndarray):

                if lower_bounds.size == 1:
                    # this is in case the value is repeated per every dimension
                    lower_bounds_arr:np.ndarray = np.tile(lower_bounds,reps=self.dimension).reshape((self.dimension,1))
                    upper_bounds_arr:np.ndarray = np.tile(upper_bounds,reps=self.dimension).reshape((self.dimension,1))

                else:
                    lower_bounds_arr:np.ndarray = lower_bounds.reshape((self.dimension,1))
                    upper_bounds_arr:np.ndarray = upper_bounds.reshape((self.dimension,1))


                # Reshape the array
                self.__bounds:np.ndarray = np.hstack(tup=(lower_bounds_arr,upper_bounds_arr))

        else:

            try:
                new_bounds = np.array(new_bounds)
            except Exception as e:
                print("Error has occured", e.args,
                      flush=True )

            if new_bounds.size == 2:
                # This is the case all the bounds are the same
                new_bounds = new_bounds.ravel()

                lower_bounds = new_bounds[0]
                upper_bounds = new_bounds[1]

                lower_bounds_arr = np.ones(shape=(self.dimension,1))*lower_bounds
                upper_bounds_arr = np.ones(shape=(self.dimension,1))*upper_bounds

                # Reshape the array
                self.__bounds:np.ndarray = np.hstack(tup=(lower_bounds_arr,upper_bounds_arr))
            
            elif np.remainder(new_bounds.size,2) == 0 and new_bounds.size > 2:

                new_bounds = new_bounds.reshape((-1,2))
                lower_bounds_arr = new_bounds[:,0].reshape((-1,1))
                upper_bounds_arr = new_bounds[:,1].reshape((-1,1))

                # Reshape the array
                self.__bounds:np.ndarray = np.hstack(tup=(lower_bounds_arr,upper_bounds_arr))
            
            else:
                raise AttributeError("The bounds should be a given in pairs", name="bounds")
    

    def compute_space_volume(self)->float:
        r"""
        This function computes the volume of the space defined by the bounds
        """
        # Compute the volume of the space
        return np.prod(self.bounds[:,1]-self.bounds[:,0])
        
   

                





    