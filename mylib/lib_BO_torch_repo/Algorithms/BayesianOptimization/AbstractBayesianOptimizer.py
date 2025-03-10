from ..AbstractAlgorithm import AbstractAlgorithm
from typing import Union, Optional, List
from pyDOE import lhs
import numpy as np
from ioh.iohcpp.problem import RealSingleObjective
from abc import abstractmethod



class LHS_sampler(object):

    __default_criteria = ("center","maximin","centermaximin","correlation")
    __reduced_criteria = {"c":"center",
                          "m":"maximin",
                          "cm":"centermaximin",
                          "corr":"correlation"}
    
    def __init__(self,
                 criterion:str="maximin",
                 iterations:int = 1000,
                 sample_zero:bool = False):
        
        self.criterion = criterion
        self.iterations = iterations
        self.sample_zero = sample_zero

    
    def __call__(self, dim:int, n_samples:int, random_seed:int=43)->np.ndarray:
        r"""
        This `__call__` overload runs the LHS Experiment and returns a `NumPy array`
        """
        
        # Set the random seed
        np.random.seed(random_seed)

        points:np.ndarray = lhs(n=dim,
                            samples=n_samples,
                            criterion=self.criterion,
                            iterations=self.iterations
                            )
        
        if self.sample_zero:
            points[0,:] = np.ones_like(points[0,:])*0.5

        

        # Return the points transformed as a Tensor
        return points.reshape((n_samples,dim))                  


    @property
    def criterion(self)->str:
        return self.__criterion
    
    @criterion.setter
    def criterion(self, new_criterion:str)->None:
        r"""
        This property holder checks if the criterion is well defined
        """

        if not isinstance(new_criterion,str):
            raise ValueError("The new criterion is not a string!")
        
        else:
            # Lower case
            new_criterion = new_criterion.lower().strip()
            if new_criterion not in self.__default_criteria:
                # Check if the criterion is in the reduced criterion
                if new_criterion in [*self.__reduced_criteria]:
                    self.__criterion = self.__reduced_criteria[new_criterion]
                else:
                    raise ValueError("The criterion is not matching the set ones!")
            else:
                self.__criterion = new_criterion
                

    @property
    def iterations(self)->int:
        return self.__iterations
    
    @iterations.setter
    def iterations(self,new_n_iter:int)->None:
        if new_n_iter > 0:
            self.__iterations = int(new_n_iter)
        else:
            raise ValueError("Negative iterations not allowed")
        
    @property
    def sample_zero(self)->bool:
        r"""
        Property for sampling the zero vector
        """
        return self.__sample_zero
    
    @sample_zero.setter
    def sample_zero(self,new_change:bool)->None:

        try:
            bool(new_change) 
        except Exception as e:
            print(e.args)
        
        # set the new value
        self.__sample_zero = new_change

class AbstractBayesianOptimizer(AbstractAlgorithm):

    
    def __init__(self, 
                 budget:int, 
                 n_DoE:Optional[int]=0,
                 random_seed:int = 43,
                 **kwargs):

        # call the initialiser from super class
        super().__init__(**kwargs)
        self.budget:int = budget
        self.n_DoE:int = n_DoE

        DoE_parameters = None
        for key,item in kwargs.items():
            if key.lower().strip() == "doe_parameters":
                DoE_parameters = item
        


        full_parameters:dict = self.__build_LHS_parameters(DoE_parameters)


        self.random_seed = random_seed

        # Check that there is some dictionary with the name "LHS_configuration"
        self.__lhs_sampler:LHS_sampler = LHS_sampler(criterion=full_parameters['criterion'],
                                                     iterations=full_parameters['iterations'],
                                                     sample_zero=full_parameters['sample_zero'])
        
        
        
        # Store all the evaluations (x,y)
        self.__x_evals:List[np.ndarray] = []
        self.__f_evals:List[float] = []
    
    def __str__(self):
        pass

    def __call__(self, problem, dim:int, bounds:np.ndarray,**kwargs)->None:

        # Call the superclass
        super().__call__(problem,
                         dim,
                         bounds,
                         **kwargs)
        
        if not isinstance(self.n_DoE,int) or self.n_DoE==0:
            # Assign this equal to the dimension of the problem
            self.n_DoE = self.dimension
        
        # Sample the points
        init_points:np.ndarray = self.lhs_sampler(self.dimension,
                                         self.n_DoE,self.random_seed)
        
        # Rescale the initial points
        init_points = self._rescale_lhs_points(init_points)
        # perform a loop with each point
        for _, point in enumerate(init_points):
            # append the new points
            self.__x_evals.append(point)
            self.__f_evals.append(problem(point))

        # Redefine the best
        self.assign_new_best()
        
        self.number_of_function_evaluations = self.n_DoE

        # Print best to screen if verbose
        if self.verbose:
            print("After Initial sampling...", f"Current Best: x:{self.__x_evals[self.current_best_index]} y:{self.current_best}"
                  ,flush=True)
                
        pass

    def _rescale_lhs_points(self,raw_lhs_points:np.ndarray):
        r"""
        This function is defined in order to take the Latin Hypercube Samples
        and project these points into the raw space defined by each dimension

        Args
        -------
        - raw_lhs_points (`np.ndarray`): A NumPy array with the initial samples coming from DoE (some points between 0 to 1)
        """

        # Take a copy of the raw points
        new_array:np.ndarray = np.empty_like(raw_lhs_points)

        # Perform a loop all over the dimensionality of the points
        for dim in range(self.dimension):

            # Compute the multiplier
            multiplier:float = self.bounds[dim,1] - self.bounds[dim,0]
            new_array[:,dim] = multiplier*raw_lhs_points[:,dim] + self.bounds[dim,0]
        
        return new_array
            
        
    
    @abstractmethod
    def assign_new_best(self):
        if self.maximisation:
            self.current_best = max(self.__f_evals)
            
        else:
            self.current_best = min(self.__f_evals)
        
        # Assign the index
        self.current_best_index = self.__f_evals.index(self.current_best, # Value
                                                       self.current_best_index  #Starting search position
                                                       )

    def __repr__(self):
        return super(AbstractAlgorithm,self).__repr__()


    def __build_LHS_parameters(self, params_dict:Union[dict,None])->dict:
        r"""
        This function builds the LHS parameters to initialize the optimisation 
        loop.
        """

        """
        samples=self.n_samples,
                            criterion=self.criterion,
                            iterations=self.iterations
        """

        complete_params_dict:dict = {"criterion":"center",
                                     "iterations":1000,
                                     "sample_zero":False}
        
        if isinstance(params_dict,dict):
            for key, item in params_dict.items():
                complete_params_dict[key] = item

        return complete_params_dict

    def reset(self)->None:
        
        # Call the superclass reset
        super().reset()

        # Reset the evaluations
        self.__x_evals:List[np.ndarray] = []
        self.__f_evals:List[float] = []

    @property
    def budget(self)->int:
        return self.__budget
    
    @budget.setter
    def budget(self,new_budget:int)->None:
        self.__budget = int(new_budget) if new_budget > 0 else None

    @property
    def n_DoE(self)->int:
        return self.__n_DoE
    
    @n_DoE.setter
    def n_DoE(self, new_n_DOE:int)->None:
        self.__n_DoE = int(new_n_DOE) if new_n_DOE >= 0 else None
    
    @property
    def lhs_sampler(self)->LHS_sampler:
        return self.__lhs_sampler


    @property
    def f_evals(self)->List[float]:
        return self.__f_evals
    
    @property
    def x_evals(self)->List[np.ndarray]:
        return self.__x_evals
    
    @property
    def random_seed(self)->int:
        return self._random_seed
    
    @random_seed.setter
    def random_seed(self,new_seed:int)->None:
        if isinstance(new_seed,int) and new_seed >=0:
            self._random_seed = new_seed


