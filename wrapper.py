import time

import numpy as np
import sys
import os
from ioh import get_problem
from ioh.iohcpp.problem import RealSingleObjective

from abc import ABC, abstractmethod

from copy import copy, deepcopy

import pathlib

from typing import Union, List, Tuple, Optional, Callable

import warnings

## TODO: ADDENDA TO HANDLE IOH Objects
from ioh.iohcpp.problem import RealSingleObjective

__set_directory__= os.path.abspath(os.path.dirname(__file__))
print(__set_directory__)



## TODO: ADDENDA TO GENERATE AN 'ABSTRACT CLASS' FOR THE OTHER METHODS TO INHERIT FROM

class Abstract_Optimizer_Wrapper(ABC):
    r"""
    This is an abstract class to avoid further repetition of methods along all the 
    upcoming wrappers.
    """

    # The library pointing to the library
    libpath:Union[pathlib.Path,str]= pathlib.Path("")

    # The base directory (which is the same as this file)
    _base_dir:pathlib.Path = pathlib.Path(__file__).parent.resolve()

    def __init__(self, 
                 func:Callable, 
                 dim:int, 
                 ub:np.ndarray, 
                 lb:np.ndarray, 
                 total_budget:Optional[int]=100, 
                 random_seed:Optional[int]=42):
        
        r"""
        This is the class initializer for the optimizer wrapper;

        Args:
        -------------------
        - func: `Callable`: A callable object which returns a function evaluation given a set of parameters.
        - dim: `int`: The dimension of the optimization problem.
        - ub: `np.ndarray`: The general upper bound of the problem.
        - lb: `np.ndarray`: The general lower bound of the problem.
        - total_budget: `Optional[int]`: The budget of evaluations of the optimizer. Set to 100 by default.
        - random_seed: `Optional[int]`: An integer denoting the initial seed of the pseudo-random number generator. By default, it is set to 42.
        """

        # Append the search library
        full_path = pathlib.Path(os.path.join(self._base_dir,self.libpath)).absolute()

        #Register
        self._register_new_library(full_path)


        # Set the primary properties
        self.func = func
        self.dim = dim
        self.total_budget = total_budget
        self.random_seed = random_seed
        self.ub = ub
        self.lb = lb
    
    @abstractmethod
    def run(self,*args,**kwargs)->None:
        r"""
        This is a method to kickstart the optimization procedure.
        """
        pass
    
    @staticmethod
    def _register_new_library(ppath:Union[pathlib.Path,str])->None:
        r"""
        This function registers the dynamic library and points the folder to point to
        the files required to run some optimization algorithm.

        Args:
        ----------
        - ppath: `Union[pathlib.Path,str]`: A composite path to point the location of the library
        """

        # Appends the library path given by parameter
        if not isinstance(ppath,pathlib.Path):
            # Modify the instance to be an instance of the library `pathlib.Path`
            ppath = pathlib.Path(ppath)
        
        if not ppath.exists():
            raise AttributeError(f"The current path doesn't exist, please check the dependency {ppath.absolute()} exists!",
                                 name="libpath",
                                 obj=ppath)
        else:
            # Append the library in case this exists
            sys.path.append(str(ppath.absolute()))

    # @property
    # def libpath(self)->str:
    #     r"""
    #     Defines the library path to add
    #     """

    #     return self.libpath.absolute()
    
    @property
    def full_libpath(self)->str:
        r"""
        Returns the complete full path to the library
        """

        return str(self._base_dir.joinpath(self.libpath).absolute())
    
    @property
    def func(self)->Callable:
        r"""
        Refers to the function object attached to the optimizer
        """

        return self.__func
    
    @func.setter
    def func(self,new_func:Callable)->None:
        r"""
        This is the setter of the `func` property.
        """

        if callable(new_func):
            # Set the function if this is the case
            self.__func = new_func

            ### TODO: This is a handler in case the instance of the problem is an IOH instance
            if issubclass(type(new_func),RealSingleObjective):
                # Write in this case the information about the problem
                self.dim = new_func.meta_data.n_variables
                self.lb = new_func.bounds.lb
                self.ub = new_func.bounds.ub
        else:
            raise AttributeError("The passed function is not a `callable`!",
                                 name="func",
                                 obj=new_func)
    
    @func.deleter
    def func(self)->bool:
        return_val:bool = True
        try:
            del self.__func
        except:
            return_val = False
            print("There is no function object attached to this class.")
        else:
            return return_val
    
    @property
    def dim(self)->int:
        r""" 
        This property refers to the dimensionality of the evaluated problem,
        and, therefore, the dimensionality handled by the algorithm.
        """

        return self.__dim
    
    @dim.setter
    def dim(self,new_dim)->None:
        r"""
        Setter of the problem dimensionality
        """

        if isinstance(new_dim,int) and new_dim > 0:
            # Change the dimension if the condition of positivity is fulfilled
            self.__dim = new_dim
        
        else:
            raise AttributeError("The dimensionality must be a positive integer",
                                 name="new_dim",
                                 obj=new_dim)
    
    @property
    def lb(self)->float:
        r"""
        The lower bound of the problem
        """

        return self.__lb 
    
    @lb.setter
    def lb(self,
           new_lb:Union[List[float],np.ndarray,float,int])->None:
        r"""
        The setter of the lower bound of the problem
        """

        if isinstance(new_lb,(np.ndarray,np.matrix,list)):
            warnings.warn("Using the first element as the lower bound!")
            new_lb = new_lb[0]
        
        self.__lb  = float(new_lb)

            

    @property
    def ub(self)->float:
        r"""
        The upper bound of the problem
        """

        return self.__ub
    

    @ub.setter
    def ub(self,
           new_ub:Union[List[float],np.ndarray,float])->None:
        r"""
        The setter of the upper bound of the problem
        """

        if isinstance(new_ub,(np.ndarray,np.matrix,list)):
            warnings.warn("Using the first element as the lower bound!")
            new_ub = new_ub[0]
        
        self.__ub  = float(new_ub)

        
    
    @property
    def total_budget(self)->int:
        r"""
        This returns the total budget as a class property
        """

        return self.__total_budget
    
    @total_budget.setter
    def total_budget(self,new_budget:int)->None:
        r"""
        This sets the total budget of evaluations of the optimizer.
        """

        if isinstance(new_budget,int) and new_budget>0:
            self.__total_budget = new_budget
        
        else:
            raise AttributeError("The total budget must be a positive integer",
                                 name="new_budget",
                                 obj=new_budget)
    
    @property
    def random_seed(self)->int:
        r"""
        Returns the random seed property given to the algorithm
        """

        return self.__random_seed 
    
    @random_seed.setter
    def random_seed(self, new_random_seed:int)->None:
        r"""
        Sets a new random seed to kickstart the algorithm
        """

        if isinstance(new_random_seed,int) and new_random_seed > -1:
            # Set the random seed if this condition is fulfilled
            self.__random_seed = new_random_seed
        
        else:
            raise AttributeError("The random seed must be a positive integer or 0",
                                 name="new_random_seed",
                                 obj= new_random_seed)
    
    @property
    def opt(self)->Union[Callable,None]:
        r"""
        A representer of the optimizer
        """

        return self.__opt
    
    @opt.setter
    def opt(self, new_opt:Callable)->None:
        r"""
        The setter of the optimizer defined for the wrapper.
        """
        self.__opt = new_opt

        
    @opt.deleter
    def opt(self)->bool:
        r"""
        The deleter object of the optimizer property
        """
        return_val:bool = True
        try:
            del self.__opt
        except:
            return_val = False
            print("There is no optimizer attached to this class.")
        else:
            return return_val
    
    ### NOTE: The following functions are only defined for Bayesian Optimizer types,
    ###       but will be defined within this abstract class for logging purposes.

    @property
    def acq_opt_time(self)->float:
        return self.get_acq_time()

    def get_acq_time(self)->float:
        r"""
        Get the time elapsed by the acquisition at each iteration of the algorithm
        """

        if self.opt is not None:
            if hasattr(self.opt,"acq_opt_time"):
                return getattr(self.opt,"acq_opt_time")
            else:
                warnings.warn("The optimizer doesn't have the `acq_opt_time` attribute")
                return 0
        
        else:
            raise ValueError("The optimizer is not set")
    
    @property
    def mode_fit_time(self)->float:
        return self.get_mode_time()
        

    def get_mode_time(self)->float:
        r"""
        Get the model fitting time
        """
    
        if self.opt is not None:
            if hasattr(self.opt,"mode_fit_time"):
                return getattr(self.opt,"mode_fit_time")
            else:
                warnings.warn("The optimizer doesn't have the `mode_fit_time` attribute")
                return 0
        
        else:
            raise ValueError("The optimizer is not set")
    
    @property
    def cum_iteration_time(self)->float:
        return self.get_iter_time()


    def get_iter_time(self)->float:
        r"""
        Get the cumulative iteration time
        """
    
        if self.opt is not None:
            if hasattr(self.opt,"cum_iteration_time"):
                return getattr(self.opt,"cum_iteration_time")
            else:
                warnings.warn("The optimizer doesn't have the `cum_iteration_time` attribute")
                return 0
        
        else:
            raise ValueError("The optimizer is not set")
    

class Abstract_Bayesian_Optimizer_Wrapper(Abstract_Optimizer_Wrapper):
    r"""
    This is a "byproduct" class to manage the Bayesian Optimization Algorithms.
    This class just adds a DoE property to define the size of the initial DoE for any corresponding
    Bayesian Optimization class algorithm.
    """

    libpath = pathlib.Path("")

    def __init__(self, 
                 func, 
                 dim, 
                 ub, 
                 lb,
                 DoE_size:Optional[int]=None,
                 sample_zero:bool = False,
                 total_budget = ..., 
                 random_seed = ...):
        
        r"""
        This is the initializer for the corresponding Bayesian Optimizers.
        """


        # Set the primary properties
        self.func = func
        self.dim = dim
        self.total_budget = total_budget
        self.random_seed = random_seed
        self.ub = ub
        self.lb = lb

        # Set the DoE Size
        if DoE_size is None:
            # Set this to the dimension
            DoE_size = dim
        
        self.Doe_size = DoE_size


        # This is a new add-on to tell the optimizer to sample a 0 vector on the LHS/Initial sampling stage
        self.sample_zero = sample_zero

        # Append the search library
        full_path = str(self._base_dir.joinpath(self.libpath))

        #Register
        self._register_new_library(full_path)
    
    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @property
    def Doe_size(self)->int:
        r"""
        This property returns the DoE Size
        """

        return self.__doe_size
    
    @Doe_size.setter
    def Doe_size(self, new_doe_size:int)->int:
        r"""
        This is the setter of the DoE Size property
        """

        if isinstance(new_doe_size,int) and new_doe_size>0:
            self.__doe_size = new_doe_size
        
        else:
            raise AttributeError("The new DoE Size must be a positive integer",
                                 name="new_doe_size",
                                 obj=new_doe_size)
    
    @property
    def sample_zero(self)->bool:
        return self.__sample_zero
    
    @sample_zero.setter
    def sample_zero(self,new_change:bool)->None:
        try:
            new_change = bool(new_change)
        except Exception as e:
            print(e.args)
        
        # Set the value
        self.__sample_zero = new_change
    



class Py_CMA_ES_Wrapper(Abstract_Optimizer_Wrapper):

    r"""
    A wrapper for CMA-ES method to compare the other methods.
    The library is the one from Nikolaus Hansen.
    """


    libpath = pathlib.Path(os.path.join('mylib', 'lib_BO_bayesoptim', 'Bayesian-Optimization'))
    def __init__(self, func, dim, ub, lb, total_budget, random_seed):
        
        # Use the superclass to initialize this wrapper
        super().__init__(func,
                         dim,
                         ub,
                         lb,
                         total_budget,
                         random_seed)

    def run(self):
        from bayes_optim import RandomForest, BO, GaussianProcess
        import cma
        from bayes_optim.extension import RealSpace

        import random
        space = RealSpace([self.lb, self.ub], random_seed=self.random_seed) * self.dim
        ma = float('-inf')
        argmax = None
        for i in range(10*self.dim):
            x = space.sample(1)[0]
        
        self.opt = cma.fmin

        self.opt(self.func, x, 1., options={'bounds': [
                 [self.lb]*self.dim, [self.ub]*self.dim], 'maxfevals': self.total_budget, 'seed': self.random_seed})



class SaasboWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    This is the wrapper of SAASBO algorithm.
    """

    libpath = pathlib.Path(os.path.join('mylib', 'lib_saasbo'))

    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Initialize the abstract super class
        super().__init__(func,
                         dim,
                         ub,
                         lb,
                         DoE_size,
                         total_budget,
                         random_seed,
                         sample_zero)

    def run(self,**kwargs):
        #from saasbo import run_saasbo, get_acq_time, get_mode_time
        from saasbo import Saasbo


        # run_saasbo(
        #     self.func,
        #     np.ones(self.dim) * self.ub,
        #     np.ones(self.dim) * self.lb,
        #     self.total_budget,
        #     self.Doe_size,
        #     self.random_seed,
        #     alpha=0.01,
        #     num_warmup=256,
        #     num_samples=256,
        #     thinning=32,
        #     device="cpu",
        # )

        # Use the other properties as given by parameter
        alpha:float = float(kwargs.pop("alpha",0.01))
        num_warmup:int = int(kwargs.pop("num_warmup",256))
        num_samples:int = int(kwargs.pop("num_samples",256))
        thinning:int = int(kwargs.pop("thinning",32))
        device:str = str(kwargs.pop("device","cpu")).lower()

        self.opt = Saasbo(func=self.func,
                          dim=self.dim,
                          ub=self.ub,
                          lb=self.lb,
                          total_budget=self.total_budget,
                          DoE_size=self.Doe_size,
                          random_seed=self.random_seed)

        print(self.opt.run_saasbo(self.func,np.ones(self.dim) * self.ub,np.ones(self.dim) * self.lb,self.total_budget,
                                  self.Doe_size,
                                  self.random_seed,
                                  alpha=alpha,
                                  num_warmup=num_warmup,
                                  num_samples=num_samples,
                                  thinning=thinning,
                                  device=device,))

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time



class BO_sklearnWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    This uses the default Bayesian Optimization defined in 
    scikit-optimize and derivatives from scikit-learn library.
    """

    libpath = pathlib.Path(os.path.join('mylib', 'lib_BO_sklearn'))

    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Initialize the abstract super class
        super().__init__(func,
                         dim,
                         ub,
                         lb,
                         DoE_size,
                         total_budget,
                         random_seed,
                         sample_zero)

    def run(self,**kwargs):
        from bosklearn import bosklearn

        acq_func:str = str(kwargs.pop("acq_func","EI"))
        noise:float = float(kwargs.pop("noise",0.1**2))

        self.opt= bosklearn(func=self.func,
                          dim=self.dim,
                          ub=self.ub,
                          lb=self.lb,
                          total_budget=self.total_budget,
                          DoE_size=self.Doe_size,
                          random_seed=self.random_seed)
        self.opt.gp_minimize(self.func,  # the function to minimize
                    # the bounds on each dimension of x
                    list((((self.lb, self.ub),) * self.dim)),
                    acq_func=acq_func,  # the acquisition function
                    n_calls=self.total_budget,  # the number of evaluations of f
                    n_random_starts=self.Doe_size,  # the number of random initialization points
                    noise=noise,  # the noise level (optional)
                    random_state=self.random_seed)

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time

class BO_bayesoptimWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r""" This is the BO of Hao Wang's library"""

    libpath = pathlib.Path(os.path.join('mylib', 'lib_' + "BO_bayesoptim"))


    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):

        # Use the super class initializer
        super().__init__(func,
                         dim,
                         ub,
                         lb,
                         DoE_size,
                         total_budget,
                         random_seed,
                         sample_zero)
        
        print(sys.path)

    def run(self):
        from bayes_optim import BO, RealSpace
        from bayes_optim.surrogate import GaussianProcess

        space = RealSpace([self.lb, self.ub]) * \
            self.dim  # create the search space

        # hyperparameters of the GPR model
        thetaL = 1e-10 * (self.ub - self.lb) * np.ones(self.dim)
        thetaU = 10 * (self.ub - self.lb) * np.ones(self.dim)
        model = GaussianProcess(  # create the GPR model
            thetaL=thetaL, thetaU=thetaU
        )

        self.opt = BO(
            search_space=space,
            obj_fun=self.func,
            model=model,
            DoE_size=self.Doe_size,  # number of initial sample points
            max_FEs=self.total_budget,  # maximal function evaluation
            verbose=True
        )
        self.opt.run()


class BO_development_bayesoptimWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    This is  the latest development from Hao Wang's Library
    """

    libpath = pathlib.Path(os.path.join('mylib', 'lib_BO_bayesoptim', 'Bayesian-Optimization'))
    # Latest changes from Hao's repository
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Use the superclass initializer
        super().__init__(func,
                         dim,
                         ub,
                         lb,
                         DoE_size,
                         total_budget,
                         random_seed,
                         sample_zero)

    @staticmethod
    def _register_new_library(libpath):
        r"""
        This is a re-interpretation of this function from the `Abstract_Optimizer_Wrapper` class.
        The modification with respect to the 'default' function is the advice on where to get the
        'Bayesian-Optimization' module.

        Args:
        ----------
        - libpath: `Union[pathlib.Path,str]`: A composite path to point the location of the library
        """

        # Appends the library path given by parameter
        if not isinstance(libpath,pathlib.Path):
            # Modify the instance to be an instance of the library `pathlib.Path`
            libpath = pathlib.Path(libpath)
        
        if not libpath.exists():
            raise AttributeError(f'No such module Bayesian-Optimization, please consider cloning this repository: https://github.com/wangronin/Bayesian-Optimization to the folder mylib/lib_BO_bayesoptim/',
                                 name="libpath",
                                 obj=libpath)
        else:
            # Append the library in case this exists
            sys.path.append(libpath)

    def run(self):
        from bayes_optim.extension import RealSpace
        from bayes_optim.bayes_opt import BO

        space = RealSpace([self.lb, self.ub], random_seed=self.random_seed) * self.dim
        self.opt = BO(
            search_space=space,
            obj_fun=self.func,
            DoE_size=self.Doe_size,
            n_point=1,
            random_seed=self.random_seed,
            acquisition_optimization={"optimizer": "BFGS"},
            max_FEs=self.total_budget,
            verbose=False,
        )
        self.opt.run()

class KPCABOWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    The KPCABO Method from Kirill and Elena (and the others).
    """

    libpath = pathlib.Path(os.path.join('mylib', 'lib_BO_bayesoptim', 'Bayesian-Optimization'))

    # Latest changes from Hao's repository
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Use the superclass initializer
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)

    @staticmethod
    def _register_new_library(libpath):
        r"""
        This is a re-interpretation of this function from the `Abstract_Optimizer_Wrapper` class.
        The modification with respect to the 'default' function is the advice on where to get the
        'Bayesian-Optimization' module.

        Args:
        ----------
        - libpath: `Union[pathlib.Path,str]`: A composite path to point the location of the library
        """

        # Appends the library path given by parameter
        if not isinstance(libpath,pathlib.Path):
            # Modify the instance to be an instance of the library `pathlib.Path`
            libpath = pathlib.Path(libpath)
        
        if not libpath.exists():
            raise AttributeError(f'No such module Bayesian-Optimization, please consider cloning this repository: https://github.com/wangronin/Bayesian-Optimization to the folder mylib/lib_BO_bayesoptim/',
                                 name="libpath",
                                 obj=libpath)
        else:
            # Append the library in case this exists
            sys.path.insert(0,str(libpath.absolute()))
        
        print(sys.path)
        
    def run(self, **kwargs):
        

        from bayes_optim import RandomForest, BO, GaussianProcess

        from bayes_optim.extension import PCABO, RealSpace, KernelPCABO, KernelFitStrategy
        from bayes_optim.mylogging import eprintf

        import random


        verbose = bool(kwargs.pop("verbose",False))
        n_point = int((kwargs.pop("n_point",1)))
        max_information_loss = float((kwargs.pop("max_information_loss",0.1)))
        acquisition_optimizer:str = str(kwargs.pop("acquisition_optimizer","BFGS"))

        space = RealSpace([self.lb, self.ub], random_seed=self.random_seed) * self.dim

        warm_data = None
        # This is to define the case to sample the zero during the DoE Stage
        if self.sample_zero:
            x_init:np.ndarray = np.zeros((1,self.dim))
            fX_init = self.func(x_init)
            warm_data = (x_init,fX_init)
            self.Doe_size = self.Doe_size-1

        self.opt = KernelPCABO(
            search_space=space,
            obj_fun=self.func,
            DoE_size=self.Doe_size,
            max_FEs=self.total_budget,
            verbose=verbose,
            n_point=n_point,
            acquisition_optimization={"optimizer": acquisition_optimizer},
            max_information_loss=max_information_loss,
            kernel_fit_strategy=KernelFitStrategy.AUTO,
            NN=self.dim,
            random_seed=self.random_seed,
            warm_data = warm_data
        )

        print(self.opt.run())

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time


class randomWrapper(Abstract_Optimizer_Wrapper):
    r"""
    A wrapper for Random Search Methods
    """

    libpath = pathlib.Path("")
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):

        self.func = func
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.total_budget = total_budget
        self.Doe_size = DoE_size
        self.random_seed = random_seed

    def run(self):
        import ioh

        def random_search(func:Callable, 
                          search_space, 
                          budget:int):
            r"""
               Implements random search to minimize a problem of size n.

               Args:
               ---------------
               - objective_function: The objective function to be minimized. It must accept a vector of size n as input and return a numeric value.
               - search_space: A list of tuples, each containing the range of allowable values for each dimension.
               - budget: The maximum number of evaluations of the objective function allowed.

               Returns:
               ---------------
               - best_solution: The best solution found.
               - best_score: The minimum value of the objective function associated with the best solution.
            """
            best_solution = None
            best_score = float('inf')

            for _ in range(budget):
                solution = [np.random.uniform(low, high) for (low, high) in search_space]
                score = self.func(solution)

                # Update the best solution if necessary
                if score < best_score:
                    best_solution = solution
                    best_score = score
                print(best_score)
            return best_solution, best_score
        search_space = [(self.lb, self.ub) for _ in range(self.dim)]
        budget = self.total_budget
        self.opt = random_search

        # Run random search
        self.opt(self.func, search_space, budget)


class linearPCABOWrapper(Abstract_Bayesian_Optimizer_Wrapper):


    libpath = pathlib.Path(os.path.join('mylib', 'lib_BO_bayesoptim', 'Bayesian-Optimization'))


    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Call the super initializer
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)

    @staticmethod
    def _register_new_library(libpath):
        r"""
        This is a re-interpretation of this function from the `Abstract_Optimizer_Wrapper` class.
        The modification with respect to the 'default' function is the advice on where to get the
        'Bayesian-Optimization' module.

        Args:
        ----------
        - libpath: `Union[pathlib.Path,str]`: A composite path to point the location of the library
        """

        # Appends the library path given by parameter
        if not isinstance(libpath,pathlib.Path):
            # Modify the instance to be an instance of the library `pathlib.Path`
            libpath = pathlib.Path(libpath)
        
        if not libpath.exists():
            raise AttributeError(f'No such module Bayesian-Optimization, please consider cloning this repository: https://github.com/wangronin/Bayesian-Optimization to the folder mylib/lib_BO_bayesoptim/',
                                 name="libpath",
                                 obj=libpath)
        else:
            # Append the library in case this exists
            sys.path.insert(0,str(libpath.absolute()))

    def run(self, **kwargs):

        from bayes_optim.extension import PCABO, RealSpace

        # Extract the kwargs to set up the setup
        verbose = bool(kwargs.pop("verbose",False))
        n_point = int((kwargs.pop("n_point",1)))
        n_components = float((kwargs.pop("n_components",0.9)))
        acquisition_optimizer:str = str(kwargs.pop("acquisition_optimizer","BFGS"))

        space = RealSpace([self.lb, self.ub]) * self.dim
        
        warm_data = None
        # This is to define the case to sample the zero during the DoE Stage
        if self.sample_zero:
            x_init:np.ndarray = np.zeros((1,self.dim))
            fX_init = self.func(x_init)
            warm_data = (x_init,fX_init)
            self.Doe_size = self.Doe_size-1

        

        
        self.opt = PCABO(
            search_space=space,
            obj_fun=self.func,
            DoE_size=self.Doe_size,
            max_FEs=self.total_budget,
            verbose=verbose,
            n_point=n_point,
            n_components=n_components,
            acquisition_optimization={"optimizer": acquisition_optimizer},
            random_seed=self.random_seed,
            warm_data = warm_data
        )
        
        print(self.opt.run())

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time


class RDUCBWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    Wrapper of RDUCBW 
    """

    libpath = pathlib.Path(os.path.join('mylib', 'lib_RDUCB/HEBO/RDUCB'))

    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        # Use the superclass initializer
        super().__init__(func,
                         dim,
                         ub,
                         lb,
                         DoE_size,
                         total_budget,
                         random_seed,
                         sample_zero)

    def run(self):
        # import sys
        # sys.path.insert(0, "./mylib/lib_linearPCABO/Bayesian-Optimization")

        from hdbo.algorithms import RDUCB
        self.opt = RDUCB( algorithm_random_seed=self.random_seed,
                    eps=-1,
                    exploration_weight= 'lambda t: 0.5 * np.log(2*t)',
                    graphSamplingNumIter=100,
                    learnDependencyStructureRate=1,
                    lengthscaleNumIter=2,
                    max_eval=-4,
                    noise_var= 0.1,
                    param_n_iter=16,
                    size_of_random_graph=0.2,
                    # data_random_seed=self.random_seed,
                    fn_noise_var=0.15,
                    grid_size=150,
                    fn= self.func,
                n_iter=self.total_budget-self.Doe_size,
                n_rand=self.Doe_size, dim=self.dim,)
        self.opt.run()


    #TODO: This is a correction applied to the method since there is a
    #      wrapper around this object.
    def get_acq_time(self):
        return self.opt.mybo.acq_opt_time

    def get_mode_time(self):
        return self.opt.mybo.mode_fit_time

    def get_iter_time(self):
        return self.opt.mybo.cum_iteration_time


class turbo1Wrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    Wrapper of turbo 1 algorithm
    """

    libpath = pathlib.Path(os.path.join('mylib', 'lib_turbo1'))

    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Call the Superclass constructor
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)
        print(sys.path)
        

    def run(self, **kwargs):

        # Load the libraries
        from turbo import Turbo1
        import torch
        import math
        import matplotlib
        import matplotlib.pyplot as plt

        # Set the specific variables
        verbose = bool(kwargs.pop("verbose",True))
        use_ard = bool(kwargs.pop("use_ard",True))
        max_cholesky_size = int(kwargs.pop("max_cholesky_size",2000))
        n_training_steps = int(kwargs.pop("n_training_steps",2000))
        min_cuda = int(kwargs.pop("min_cuda",2000))
        device = str(kwargs.pop("device","cpu"))
        dtype = str(kwargs.pop("dtype","float64"))


        self.opt = Turbo1(
            f=self.func,  # Handle to objective function
            lb=np.ones(self.dim) * self.lb,  # Numpy array specifying lower bounds
            ub=np.ones(self.dim) * self.ub,  # Numpy array specifying upper bounds
            n_init=self.Doe_size,  # Number of initial bounds from an Latin hypercube design
            max_evals=self.total_budget,  # Maximum number of evaluations
            batch_size=5,  # How large batch size TuRBO uses
            verbose=verbose,  # Print information from each batch
            use_ard=use_ard,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=max_cholesky_size,  # When we switch from Cholesky to Lanczos
            n_training_steps=n_training_steps,  # Number of steps of ADAM to learn the hypers
            min_cuda=min_cuda,  # Run on the CPU for small datasets
            device=device,  # "cpu" or "cuda"
            dtype=dtype,  # float64 or float32
            sample_zero = self.sample_zero # Get to sample zero as part of the function evaluations
        )
        self.opt.optimize()

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time


class turbomWrapper(Abstract_Bayesian_Optimizer_Wrapper):

    libpath = pathlib.Path(os.path.join('mylib', 'lib_turbo1'))


    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Call the Superclass constructor
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)
        print(sys.path)
        

    def run(self,**kwargs) :
        from turbo import TurboM
        import torch
        import math
        import matplotlib
        import matplotlib.pyplot as plt
        #tr = math.floor(self.total_budget / self.Doe_size) - 1

        # Set the specific variables
        verbose = bool(kwargs.pop("verbose",True))
        use_ard = bool(kwargs.pop("use_ard",True))
        max_cholesky_size = int(kwargs.pop("max_cholesky_size",2000))
        n_training_steps = int(kwargs.pop("n_training_steps",2000))
        min_cuda = int(kwargs.pop("n_training_steps",2000))
        batch_size = int(kwargs.pop("batch_size",5))
        device = str(kwargs.pop("device","cpu"))
        dtype = str(kwargs.pop("dtype","float64"))

        tr = max(int(self.dim/5),2)
        n_init = math.floor(self.Doe_size/tr)


        self.opt = TurboM(
            f=self.func,  # Handle to objective function
            lb=np.ones(self.dim) * self.lb,  # Numpy array specifying lower bounds
            ub=np.ones(self.dim) * self.ub,  # Numpy array specifying upper bounds
            n_init=n_init,  # Number of initial bounds from an Symmetric Latin hypercube design
            max_evals=self.total_budget,  # Maximum number of evaluations
            n_trust_regions=tr,  # Number of trust regions
            batch_size=batch_size,  # How large batch size TuRBO uses
            verbose=verbose,  # Print information from each batch
            use_ard=use_ard,  # Set to true if you want to use ARD for the GP kernel
            max_cholesky_size=max_cholesky_size,  # When we switch from Cholesky to Lanczos
            n_training_steps=n_training_steps,  # Number of steps of ADAM to learn the hypers
            min_cuda=min_cuda,  # Run on the CPU for small datasets
            device=device,  # "cpu" or "cuda"
            dtype=dtype,  # float64 or float32
            sample_zero= self.sample_zero # Switch to sample zero as part of the DoE
        )
        self.opt.optimize()

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time

class EBOWrapper(Abstract_Bayesian_Optimizer_Wrapper):

    libpath = pathlib.Path(os.path.join('mylib', 'lib_EBO'))


    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed, sample_zero):
         # import sys
                 # sys.path.append('./mylib/' + 'lib_' + "EBO")
                         # print(sys.path)

        # Call the Superclass constructor
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)
        print(sys.path)

    def run(self, **kwargs):
        import numpy.matlib as nm
        import functools
        from ebo_core.ebo import ebo
        from test_functions.simple_functions import sample_z
        import time
        import logging

        dx = self.dim
        z = sample_z(dx)
        k = np.array([10] * dx)
        x_range = nm.repmat([[self.lb], [self.ub]], 1, self.dim)
        x_range = x_range.astype(float)
        sigma = 0.01
        n = self.Doe_size
        budget = self.total_budget
        f = self.func
        f = functools.partial(lambda f, x: -f(x), f)
        options = {'x_range': x_range,  # input domain
                  'dx': x_range.shape[1],  # input dimension
                  'max_value': 0,  # target value
                  'T': budget,  # number of iterations
                  'B': 1,  # number of candidates to be evaluated
                  'dim_limit': 3,  # max dimension of the input for each additive function component
                  'isplot': 0,  # 1 if plotting the result; otherwise 0.
                  'z': None, 'k': None,  # group assignment and number of cuts in the Gibbs sampling subroutine
                  'alpha': 1.,  # hyperparameter of the Gibbs sampling subroutine
                  'beta': np.array([5., 2.]),
                  'opt_n': 1000,  # points randomly sampled to start continuous optimization of acfun
                  'pid': 'test3',  # process ID for Azure
                  'datadir': 'tmp_data/',  # temporary data directory for Azure
                  'gibbs_iter': 10,  # number of iterations for the Gibbs sampling subroutine
                  'useAzure': False,  # set to True if use Azure for batch evaluation
                  'func_cheap': True,  # if func cheap, we do not use Azure to test functions
                  'n_add': None,  # this should always be None. it makes dim_limit complicated if not None.
                  'nlayers': 100,  # number of the layers of tiles
                  'gp_type': 'l1',  # other choices are l1, sk, sf, dk, df
                  'gp_sigma': 0.1,  # noise standard deviation
                  'n_bo': 10,  # min number of points selected for each partition
                  'n_bo_top_percent': 0.5,  # percentage of top in bo selections
                  'n_top': 10,  # how many points to look ahead when doing choose Xnew
                  'min_leaf_size': 10,  # min number of samples in each leaf
                  'max_n_leaves': 10,  # max number of leaves
                  'thresAzure': 1,  # if batch size > thresAzure, we use Azure
                  'save_file_name': 'tmp/tmp.pk',
                  }
        self.opt = ebo(f, options)
        start = time.time()
        self.opt.run()

        print("elapsed time: ", time.time() - start)

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time

class EBO_BWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    
    libpath = pathlib.Path(os.path.join('mylib', 'lib_EBO'))


    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed,sample_zero):
        
        # Call the super class initializer
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)


    def run(self, **kwargs):
        import numpy.matlib as nm
        import functools
        from ebo_core.ebo import ebo
        from test_functions.simple_functions import sample_z
        import time
        import logging

        dx = self.dim
        z = sample_z(dx)
        k = np.array([10] * dx)
        x_range = nm.repmat([[self.lb], [self.ub]], 1, self.dim)
        x_range = x_range.astype(float)
        sigma = 0.01
        n = self.Doe_size
        budget = self.total_budget
        t= int (float(budget)/10)

        f = self.func
        f = functools.partial(lambda f, x: -f(x), f)

        options = {'x_range': x_range,  # input domain
                  'dx': x_range.shape[1],  # input dimension
                  'max_value': 0,  # target value
                  'T': t,  # number of iterations
                  'B': 10,  # number of candidates to be evaluated
                  'dim_limit': 3,  # max dimension of the input for each additive function component
                  'isplot': 0,  # 1 if plotting the result; otherwise 0.
                  'z': None, 'k': None,  # group assignment and number of cuts in the Gibbs sampling subroutine
                  'alpha': 1.,  # hyperparameter of the Gibbs sampling subroutine
                  'beta': np.array([5., 2.]),
                  'opt_n': 1000,  # points randomly sampled to start continuous optimization of acfun
                  'pid': 'test3',  # process ID for Azure
                  'datadir': 'tmp_data/',  # temporary data directory for Azure
                  'gibbs_iter': 10,  # number of iterations for the Gibbs sampling subroutine
                  'useAzure': False,  # set to True if use Azure for batch evaluation
                  'func_cheap': True,  # if func cheap, we do not use Azure to test functions
                  'n_add': None,  # this should always be None. it makes dim_limit complicated if not None.
                  'nlayers': 100,  # number of the layers of tiles
                  'gp_type': 'l1',  # other choices are l1, sk, sf, dk, df
                  'gp_sigma': 0.1,  # noise standard deviation
                  'n_bo': 10,  # min number of points selected for each partition
                  'n_bo_top_percent': 0.5,  # percentage of top in bo selections
                  'n_top': 10,  # how many points to look ahead when doing choose Xnew
                  'min_leaf_size': 10,  # min number of samples in each leaf
                  'max_n_leaves': 10,  # max number of leaves
                  'thresAzure': 1,  # if batch size > thresAzure, we use Azure
                  'save_file_name': 'tmp/tmp.pk',
                  }
        self.opt = ebo(f, options)
        start = time.time()
        self.opt.run()

        print("elapsed time: ", time.time() - start)

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time

class ALEBOWrapper(Abstract_Bayesian_Optimizer_Wrapper):

    r"""
    Wrapper of the ALEBO Method
    """
    libpath = pathlib.Path("")
    
    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed):
        # Call the Super Class
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)

    def run(self,**kwargs):
        # import pathlib
        # my_dir = pathlib.Path(__file__).parent.resolve()
        # sys.path.append(os.path.join(my_dir, 'mylib', 'lib_ALEBO'))
        # import numpy as np
        from ax.utils.measurement.synthetic_functions import branin
        print(sys.path)
        
        def branin_evaluation_function(parameterization):
            # Evaluates Branin on the first two parameters of the parameterization.
            # Other parameters are unused.
            x = np.array([parameterization["x0"], parameterization["x1"]])

            return {"objective": (branin(x), 0.0)}
        
        def function(parameterization):
            # Evaluates Branin on the first two parameters of the parameterization.
            # Other parameters are unused.
            x = np.array([parameterization[f'x{i}'] for i in range(self.dim)])
            self.iter+=1
            if self.iter == self.total_budget:
                print("Optimization is complete, cannot run another trial.")
                exit()
            return {"objective": (self.func(x), 0.0)}
        
        parameters = [
            {"name": "x0", "type": "range", "bounds": [self.lb, self.ub], "value_type": "float"},
            {"name": "x1", "type": "range", "bounds": [self.lb, self.ub], "value_type": "float"},
        ]
        parameters.extend([
            {"name": f"x{i}", "type": "range", "bounds": [self.lb, self.ub], "value_type": "float"}
            for i in range(2, self.dim)
        ])
        from ax.modelbridge.strategies.alebo import ALEBOStrategy
        alebo_strategy = ALEBOStrategy(D=self.dim, d=4, init_size=self.Doe_size)
        alebo_strategy._steps[0].model_kwargs.update({"seed": self.random_seed})
        from ax.service.managed_loop import optimize
        self.opt = optimize
        
        # Call the optimizer
        self.opt(parameters=parameters,
        experiment_name="test",
        objective_name="objective",
        evaluation_function=function,
        minimize=True,
        total_trials=self.total_budget,
        generation_strategy=alebo_strategy,
        )

#     def run(self):
#         # import sys
#         # sys.path.insert(0, "./mylib/lib_linearPCABO/Bayesian-Optimization")
#         parameters = [
#             {"name": "x0", "type": "range", "bounds": [self.lb, self.ub], "value_type": "float"},
#             {"name": "x1", "type": "range", "bounds": [self.lb, self.ub], "value_type": "float"},
#         ]
#         parameters.extend([
#             {"name": f"x{i}", "type": "range", "bounds": [self.lb, self.ub], "value_type": "float"}
#             for i in range(2, self.dim)
#         ])
#         alebo_strategy = ALEBOStrategy(D=self.dim, d=10, init_size=self.Doe_size)
#         from ax.service.managed_loop import optimize
#         self.opt = optimize(
#     parameters=parameters,
#     experiment_name="test",
#     objective_name="objective",
#     evaluation_function=self.func,
#     minimize=True,
#     total_trials=self.total_budget,
#     generation_strategy=alebo_strategy,
# )
#         self.opt()

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time

class HEBOWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    Wrapper of the HEBO Method
    """
    libpath = pathlib.Path("")

    def __init__(self, func, dim, ub, lb, total_budget, DoE_size, random_seed, sample_zero):
 
        # Call the Super Class
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         random_seed=random_seed,
                         sample_zero=sample_zero)

    def run(self,**kwargs):

        import pandas as pd
        import numpy as np
        from hebo.design_space.design_space import DesignSpace
        from hebo.optimizers.hebo import HEBO
        # def obj(params: pd.DataFrame) -> np.ndarray:
        #     return ((params.values - 0.37) ** 2).sum(axis=1).reshape(-1, 1)
        def obj(params: pd.DataFrame) -> np.ndarray:
            # Imports the desired BBOB function (for example, function #1:sphere) 
            problem = self.func  # parameters: dimensions, index of corresponding BBOB function (1-24)

            # Calcola il valore della funzione obiettivo per ciascuna riga dei parametri
            values = [problem(np.squeeze(row.values)) for _, row in params.iterrows()]

            # Restituisci i valori come un array numpy
            return np.array(values).reshape(-1, 1)

        dimension_specs = [{"name": f"param{i}", "type": "num", 'lb' : self.lb, 'ub' : self.ub } for i in
                           range(1, self.dim + 1)]
        space = DesignSpace().parse(dimension_specs)
        #space = DesignSpace().parse([{'name': 'x', 'type': 'int', 'lb': self.lb, 'ub': self.ub}])
        self.opt:HEBO = HEBO(space, 
                        rand_sample=self.Doe_size, 
                        scramble_seed=self.random_seed )
        
        rec = self.opt.suggest(n_suggestions=self.Doe_size)
        # Fit initially the model 
        if self.sample_zero:
            # Perform a loop to rewrite the first point from DoE
            for idx in range(len(rec.param1)):
                rec.param1[idx] = 0.0
        
        # Observe the DoE
        self.opt.observe(rec, obj(rec))
        
        for i in range(self.total_budget-self.Doe_size):
            rec = self.opt.suggest(n_suggestions=1)
            self.opt.observe(rec, obj(rec))
            print('After %d iterations, best obj is %.2f' % (i, self.opt.y.min()))
            self.opt.cum_iteration_time = time.process_time()

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time
    
class BAxUSWrapper(Abstract_Bayesian_Optimizer_Wrapper):
    
    r"""
    Wrapper of the BAxUS Method
    """
    libpath = pathlib.Path(os.path.join("mylib","lib_BAxUS","BAxUS"))
    def __init__(self, func, 
                 dim:int, ub, lb, 
                 total_budget:int, 
                 DoE_size, 
                 random_seed,
                 sample_zero=False,
                 verbose=False,dtype='float64'):

        # Use the superclass constructor
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         total_budget=total_budget,
                         DoE_size=DoE_size,
                         random_seed=random_seed,
                         sample_zero=sample_zero)

        self._verbose:bool = verbose
        self._dtype:str = dtype

    def run(self, **kwargs):

        try:
            #from mylib.lib_BAxUS.BAxUS.baxus import BAxUS # Import the BAxUS object
            from baxus import BAxUS # Import the BAxUS object
            #from mylib.lib_BAxUS.BAxUS.baxus.util.behaviors import BaxusBehavior
            from baxus.util.behaviors import BaxusBehavior
            #from mylib.lib_BAxUS.BAxUS.baxus.util.behaviors.gp_configuration import GPBehaviour
            from baxus.util.behaviors.gp_configuration import GPBehaviour
        except ModuleNotFoundError as e:
            print("The module was not found!, please download it", e.args)

        # import the parser function from the BAxUS library
        #from mylib.lib_BAxUS.BAxUS.baxus.util.parsing import parse
        from baxus.util.parsing import parse
        #from mylib.lib_BAxUS.BAxUS.baxus.util.parsing import (embedding_type_mapper, 
        #                                                      acquisition_function_mapper,
        #                                                      mle_optimization_mapper)

        from baxus.util.parsing import (embedding_type_mapper, 
                                        acquisition_function_mapper,
                                        mle_optimization_mapper,
                                        fun_mapper)
        
        #from mylib.lib_BAxUS.BAxUS.baxus.benchmark_runner import fun_mapper, info
        from baxus.benchmark_runner import fun_mapper, info
        from wrapper_helper import IOH_BAxUS_Wrapper
        import logging


        import json
        import logging
        import os
        import sys
        from datetime import datetime
        from logging import info, warning
        from typing import List
        from zlib import adler32


        from baxus.util.exceptions import ArgumentError
        from baxus.util.utils import star_string

        #FORMAT = "%(asctime)s %(levelname)s: %(filename)s: %(message)s"
        #DATEFORMAT = '%m/%d/%Y %I:%M:%S %p'

        # Call the default configuration parser 
        parser_args:str = f"-f ioh_function --algorithm baxus"\
                            f" --input-dim {self.dim} --target-dim {self.dim} --n-init {self.Doe_size}"\
                            f" --max-evals {self.total_budget}"
        
        args = parse(parser_args.split())

        # directory = os.path.join(
        #     args.results_dir,
        #     f"{datetime.now().strftime('%d_%m_%Y')}{f'-{args.run_description}' if len(args.run_description) > 0 else ''}",
        # )
        # os.makedirs(directory, exist_ok=True)
        # logging.basicConfig(
        #     filename=os.path.join(directory, "logging.log"),
        #     level=logging.INFO if not args.verbose else logging.DEBUG,
        #     format=FORMAT,
        #     force=True,
        #     datefmt=DATEFORMAT
        # )

        #sysout_handler = logging.StreamHandler(sys.stdout)
        #sysout_handler.setFormatter(logging.Formatter(fmt=FORMAT, datefmt=DATEFORMAT))
        #logging.getLogger().addHandler(sysout_handler)

        #repetitions = list(range(args.num_repetitions))

        #args_dict = vars(args)
        # with open(os.path.join(directory, "conf.json"), "w") as f:
        #     f.write(json.dumps(args_dict))

        ### NOTE: Rewriting the definition of the function as:


        ###
        bin_sizing_method = embedding_type_mapper[args.embedding_type]

        acquisition_function = acquisition_function_mapper[args.acquisition_function]

        mle_optimization_method = mle_optimization_mapper[args.mle_optimization]

        input_dim = args.input_dim
        target_dim = args.target_dim

        n_init = args.n_init
        #n_init = self.Doe_size

        max_evals = args.max_evals
        noise_std = args.noise_std

        new_bins_on_split = args.new_bins_on_split

        ## +++++++++++++++++++++++++++++++++++++++++++++++++++
        ## NOTE This handler is in case for small dimensions
        if input_dim == 3:
            new_bins_on_split = 2
        elif input_dim == 2 or input_dim == 1:
            new_bins_on_split = 2

        ## ++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        multistart_samples = args.multistart_samples
        mle_training_steps = args.mle_training_steps
        multistart_after_samples = args.multistart_after_sample
        l_init = args.initial_baselength
        l_min = args.min_baselength
        l_max = args.max_baselength
        adjust_initial_target_dim = True # args.adjust_initial_target_dimension
        #print(adjust_initial_target_dim)
        budget_until_input_dim = args.budget_until_input_dim

        combs = {}

        if n_init is None:
            n_init = target_dim + 1
        if args.min_baselength > args.max_baselength:
            raise ArgumentError(
                "Minimum baselength has to be larger than maximum baselength."
            )
        if args.input_dim < args.target_dim:
            raise ArgumentError(
                "Input dimension has to be larger than target dimension."
            )
        if args.noise_std < 0:
            raise ArgumentError("Noise standard deviation has to be positive.")
        if max_evals < budget_until_input_dim:
            raise ArgumentError("budget_until_input_dim has to be <= max_evals.")
        if args.multistart_samples < 1:
            raise ArgumentError("Number of multistart samples has to be >= 1.")
        if args.multistart_after_sample > args.multistart_samples:
            raise ArgumentError(
                f"Number of multistart samples after sampling {args.multistart_after_sample} has to be smaller or equal to the numbers"
                f"of initial multistart samples {args.multistart_samples}."
            )
        if args.multistart_after_sample < 1:
            raise ArgumentError(
                "Number of multistart samples after sampling has to be >= 1."
            )
        if args.mle_training_steps < 0:
            raise ArgumentError("Number of mle training steps has to be >= 0.")
        if new_bins_on_split < 2:
            raise ArgumentError("Number of new bins on split has to be greater than one.")

        funs = {
            k: v(dim=input_dim, noise_std=noise_std)
            for k, v in fun_mapper().items()
            if k == args.function
        }

        c = {
            f"{k}_in_dim_{v.dim}_t_dim{target_dim}_n_init_{n_init}"
            f"{f'_noise_{noise_std}' if noise_std > 0 else ''}": {
                "input_dim": v.dim,
                "target_dim": min(v.dim, target_dim),
                "n_init": n_init,
                "f": v,
                "lb": v.lb_vec,
                "ub": v.ub_vec,
            }
            for k, v in funs.items()
        }

        combs.update(c)

        for i, (k, comb) in enumerate(combs.items()):
            info(f"running combination {k}")
            llb = comb["lb"]
            uub = comb["ub"]
            input_dim = comb["input_dim"]
            target_dim = comb["target_dim"]
            n_init = comb["n_init"]

            #f = comb["f"]
            f = IOH_BAxUS_Wrapper(self.func,
                                  self.dim,
                                  self.lb,
                                  self.ub,
                                  None)

            #function_dir = os.path.join(directory, k)
            #os.makedirs(function_dir, exist_ok=True)

            if "baxus" == args.algorithm:
                # *** BAxUS ***
                info("*** BAxUS***")
                behavior = BaxusBehavior(
                    n_new_bins=new_bins_on_split,
                    initial_base_length=l_init,
                    min_base_length=l_min,
                    max_base_length=l_max,
                    acquisition_function=acquisition_function,
                    embedding_type=bin_sizing_method,
                    adjust_initial_target_dim=adjust_initial_target_dim,
                    noise=noise_std,
                    budget_until_input_dim=budget_until_input_dim
                )
                gp_behaviour = GPBehaviour(
                    mll_estimation=mle_optimization_method,
                    n_initial_samples=multistart_samples,
                    n_best_on_lhs_selection=multistart_after_samples,
                    n_mle_training_steps=mle_training_steps,
                )
                # conf_name = (
                #     f"baxus_{behavior}_{gp_behaviour}"
                # )
                # run_dir = os.path.join(
                #     function_dir,
                #     str(adler32(conf_name.encode("utf-8"))),
                # )
                self.opt = BAxUS(
                    f=f,  # Handle to objective function
                    n_init=n_init,  # Number of initial bounds from an Latin hypercube design
                    max_evals=max_evals,  # Maximum number of evaluations
                    target_dim=target_dim,
                    #run_dir=run_dir,
                    #conf_name=conf_name,
                    behavior=behavior,
                    gp_behaviour=gp_behaviour,
                    sample_zero = self.sample_zero
                )
                self.opt.optimize()
                #del baxus


        # def obj(params: pd.DataFrame) -> np.ndarray:
        #     return ((params.values - 0.37) ** 2).sum(axis=1).reshape(-1, 1)
        """ def obj(params: pd.DataFrame) -> np.ndarray:
            # Imposta la funzione BBOB desiderata (ad esempio, la funzione 1)
            problem = self.func  # Parametri: dimensione, funzione BBOB (1-24)

            # Calcola il valore della funzione obiettivo per ciascuna riga dei parametri
            values = [problem(np.squeeze(row.values)) for _, row in params.iterrows()]

            # Restituisci i valori come un array numpy
            return np.array(values).reshape(-1, 1)

        dimension_specs = [{"name": f"param{i}", "type": "num", 'lb' : self.lb, 'ub' : self.ub } for i in
                           range(1, self.dim + 1)]
        space = DesignSpace().parse(dimension_specs)
        #space = DesignSpace().parse([{'name': 'x', 'type': 'int', 'lb': self.lb, 'ub': self.ub}])
        self.opt = HEBO(space, rand_sample=self.Doe_size, scramble_seed=self.random_seed )
        for i in range(self.total_budget):
            rec = self.opt.suggest(n_suggestions=1)
            self.opt.observe(rec, obj(rec))
            print('After %d iterations, best obj is %.2f' % (i, self.opt.y.min()))
            self.opt.cum_iteration_time = time.process_time() """
        
        # Now the requirement is to wrap the function from the IOH workspace
        

    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time



class SMACWrapper(Abstract_Bayesian_Optimizer_Wrapper):

    #source_lib = os.path.join(__set_directory__,"mylib","lib_BAxUS","BAxUS")
    libpath = pathlib.Path("")
    def __init__(self, func, 
                 dim:int, ub, lb, 
                 total_budget:int, 
                 DoE_size:int, 
                 random_seed:int,
                 sample_zero=False,
                 verbose=False):
        # import sys
        # sys.path.append('./mylib/' + 'lib_' + "linearPCABO")
        # print(sys.path)
        # import pathlib
        # my_dir = pathlib.Path(__file__).parent.resolve()
        # sys.path.append(os.path.join(my_dir, 'mylib', 'lib_RDUCB/HEBO/RDUCB'))
        # print(sys.path)
        # sys.path.insert(0, bayes_bo_lib)
        # print(sys.path)

        # Use the SuperClass Constructor
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         DoE_size=DoE_size,
                         total_budget=total_budget,
                         sample_zero=sample_zero,
                         random_seed=random_seed)

        self._verbose:bool = verbose

    def run(self):
        # import sys
        # sys.path.insert(0, "./mylib/lib_linearPCABO/Bayesian-Optimization")
        
        from ConfigSpace import Configuration, ConfigurationSpace, Float
        from typing import Callable, List

        from smac import HyperparameterOptimizationFacade, Scenario
        from smac.runhistory.dataclasses import TrialValue, TrialInfo
        from smac.initial_design.latin_hypercube_design import LatinHypercubeInitialDesign


        class IOH_SMAC_Wrapper:

            def __init__(self,
                         f:Callable,
                         dim:int,
                         lb:float,
                         ub:float,
                         seed:int = 0):
                
                # Assign the values
                self.f = f
                self.dim = dim
                self.lb = lb
                self.ub = ub
                self.seed = seed

            @property
            def configspace(self) -> ConfigurationSpace:

                # Initialize the configuration space
                cs = ConfigurationSpace(seed=self.seed)

                list_of_variables:List[Float] = []

                for idx_ in range(self.dim):
                    list_of_variables.append(Float(f"x{str(idx_)}", (self.lb,self.ub), default= 0))

                cs.add(list_of_variables)

                return cs

            def train(self, config: Configuration, seed: int = 0) -> float:
                r"""
                This is to follow the same "lexicon" recommended by the library, which
                states that a custom made model should have this train function.
                """

                list_of_values:list = []

                for idx in range(len(config)):
                    list_of_values.append(config[f"x{idx}"])

                # Compute the `cost`
                cost = self.f(list_of_values)
                return cost
        
        def return_values_to_zero(config_space:ConfigurationSpace)->dict:
            r"""
            This function changes all the values of the present configuration to zero
            """

            config_space_names = config_space.get_default_configuration()
            



        model = IOH_SMAC_Wrapper(f=self.func,
                                 dim=self.dim,
                                 lb=self.lb,
                                 ub=self.ub,
                                 seed=self.random_seed)

        # Scenario object
        scenario = Scenario(model.configspace, deterministic=False, n_trials=self.total_budget)

        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1,  # We basically use one seed per config only
        )

        # Initialize a LHS initial sampler
        lhs_des_obj = LatinHypercubeInitialDesign(scenario=scenario,
                                                  n_configs=self.Doe_size,
                                                  seed=self.random_seed)

        # Now we use SMAC to find the best hyperparameters
        self.opt:HyperparameterOptimizationFacade = HyperparameterOptimizationFacade(
            scenario,
            model.train,
            intensifier=intensifier,
            initial_design=lhs_des_obj,
            overwrite=True,
        )

        # We can ask SMAC which trials should be evaluated next
        # Optimize the loop (PENDING THE DOE)
        for i in range(self.total_budget):
            info = self.opt.ask()

            if i==0 and self.sample_zero:
           
                new_config = model.configspace.get_default_configuration()
                
                
                
                info = TrialInfo(config=new_config,
                                 instance=info.instance,
                                 seed=info.seed,
                                 budget=info.budget)

        
            assert info.seed is not None

            cost = model.train(info.config, seed=info.seed)
            value = TrialValue(cost=cost, time=0.5)

            self.opt.tell(info, value)

        # After calling ask+tell, we can still optimize
        # Note: SMAC will optimize the next 90 trials because 10 trials already have been evaluated
        #incumbent = smac.optimize()


    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time

class REMBOWrapper(Abstract_Bayesian_Optimizer_Wrapper):

    libpath = pathlib.Path(os.path.join("mylib","lib_REMBO","HesBO"))

    def __init__(self, func, 
                 dim:int, ub, lb, 
                 total_budget:int, 
                 DoE_size:int, 
                 random_seed:int,
                 sample_zero:bool = False,
                 verbose=False):

        # Use the Super Class constructor
        super().__init__(func=func,
                         dim=dim,
                         ub=ub,
                         lb=lb,
                         total_budget=total_budget,
                         DoE_size=DoE_size,
                         random_seed=random_seed,
                         sample_zero=sample_zero)

        self._verbose:bool = verbose

    def run(self,**kwargs):


        from typing import Callable, Tuple
        try:
            from mylib.lib_REMBO.HesBO.REMBO import RunRembo2, RemboSetter
        
        except ModuleNotFoundError as e:
            print("The module was not found, importing with different way!")
            from REMBO import RunRembo2, RemboSetter
        
        # Extract the kwargs to set up REMBO parameterization
        matrix_type:str = str(kwargs.pop("matrix_type",'simple'))
        kern_inp_type:str = str(kwargs.pop("kern_inp_type",'psi'))
        ard:bool = bool(kwargs.pop("ARD",True))
        variance:float = float(kwargs.pop("variance",1.0))
        low_dim:int = int(kwargs.pop("low_dim",max(2,int(self.dim/5))))
        hyper_opt_interval:int = int(kwargs.pop("hyper_opt_interval",2))
        box_size:float = float(kwargs.pop("box_size",np.sqrt(self.dim)))

        # Define a wrapper class for the IOH_Instance
        class IOH_REMBO_Wrapper:
            def __init__(self, 
                         func:Callable, 
                         dim:int, 
                         lb:float,
                         ub:float, 
                         act_var=None, 
                         noise_var=0):
                
                self.range=np.array([(lb,ub)*dim]).reshape((-1,2))
                self.__act_var=act_var
                self.__var = noise_var
                # Include the function
                self.func = func
                self.__dim = dim

            def scale_domain(self,x)->np.ndarray:
                # Scaling the domain
                x_copy = np.copy(x)
                if len(x_copy.shape) == 1:
                    x_copy = x_copy.reshape((1, x_copy.shape[0]))

                for i in range(len(self.range)):
                    x_copy[:, i] = x_copy[:, i] * (self.range[i,1] - self.range[i,0]) / 2 + (
                            self.range[i,1] + self.range[i,0]) / 2
                    
                new_ar = np.unique(x_copy,axis=0)
                return new_ar
            
            def evaluate_true(self,x):
                x_scaled = self.scale_domain(x)
                
                f = []

                for idx_,val in enumerate(x_scaled):
                    f.append(self.func(val))

                return np.multiply(-1,np.array(f).reshape((-1,1)))

            def evaluate(self, x):
                
                x_org = self.evaluate_true(x)
                n = len(x_org)

                return x_org + np.random.normal(0,self.var,(n,1))
            

            @property
            def act_var(self)->np.ndarray:
                return self.__act_var
            
            @act_var.setter
            def act_var(self,new_act_var:np.ndarray)->None:
                self.__act_var = new_act_var
            
            @property
            def var(self)->float:
                return self.__var
            
            @var.setter
            def var(self,new_var)->None:
                self.__var = new_var


        f_wrapper:IOH_REMBO_Wrapper = IOH_REMBO_Wrapper(func=self.func,
                                                        dim = self.dim,
                                                        lb=self.lb,
                                                        ub= self.ub,
                                                        noise_var=0)
        
        self.opt:RemboSetter = RemboSetter(self.random_seed,
                                           sample_zero=self.sample_zero)
        # Call the runner
        self.opt.optimize(func=f_wrapper,
                 low_dim=low_dim, 
                 high_dim=self.dim, 
                 initial_n=self.Doe_size,
                 total_itr=self.total_budget-self.Doe_size,
                 hyper_opt_interval=hyper_opt_interval,
                 matrix_type=matrix_type,
                 kern_inp_type=kern_inp_type,
                 ARD=ard,variance=variance,
                 box_size=box_size,
                 noise_var=0)
                


    # def get_acq_time(self):
    #     return self.opt.acq_opt_time

    # def get_mode_time(self):
    #     return self.opt.mode_fit_time

    # def get_iter_time(self):
    #     return self.opt.cum_iteration_time
    

class BO_Torch_VanillaBO_Wrapper(Abstract_Bayesian_Optimizer_Wrapper):
    r"""
    This is a wrapper to use a basic code for Vanilla BO by using the most basic 
    BO-Torch implementation.
    """
    libpath = pathlib.Path(os.path.join("mylib","lib_BO_torch_repo"))


    def __init__(self, func, 
                 dim:int, ub, lb, 
                 total_budget:int, 
                 DoE_size:int, 
                 random_seed:int,
                 sample_zero:bool=False,
                 verbose=True):

        # Use the general class initializer
        super().__init__(func=func, 
                         dim=dim, 
                         ub=ub, 
                         lb=lb, DoE_size=DoE_size, 
                         total_budget=total_budget, 
                         random_seed=random_seed, 
                         sample_zero=sample_zero)

        # Set an additional verbose property
        self._verbose = verbose
    

    def run(self,**kwargs):
        # Call the Algorithm
        try:
            from mylib.lib_BO_torch_repo.Algorithms import Vanilla_BO
        except ModuleNotFoundError:
            # Call the normal way
            from Algorithms import Vanilla_BO

        
        # Extract the kwargs to set up the experiments
        acq_func:str = str(kwargs.pop("acquisition_function","EI"))
        beta:float = float(kwargs.pop("beta",0.2))

        
        # Set up the algorithm
        self.opt = Vanilla_BO(budget=self.total_budget,
                              n_DoE=self.Doe_size,
                              random_seed=self.random_seed,
                              acquisition_function=acq_func,
                              verbose = self._verbose,
                              DoE_parameters = {'sample_zero':self.sample_zero,
                                                'criterion':"center"}
                              )
        
        self.opt(problem = self.func,
                 dim = self.dim,
                 bounds = np.asarray([(self.lb,self.ub) for _ in range(self.dim)]),
                 beta=beta)



def wrapopt(optimizer_name:str, 
            func:Callable, 
            ml_dim:int, 
            ml_total_budget:int,
            ml_DoE_size:int, 
            random_seed:int, 
            sample_zero:bool=False):
    ub = +5
    lb = -5
    if optimizer_name == "saasbo":
        return SaasboWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == "BO_sklearn":
        return BO_sklearnWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                                 random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == "BO_bayesoptim":
        return BO_bayesoptimWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                                    random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == "random":
        return randomWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed)
    if optimizer_name == "linearPCABO":
        return linearPCABOWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == "turbo1":
        return turbo1Wrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == "turbom":
        return turbomWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == 'BO_dev_Hao':
        return BO_development_bayesoptimWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == 'EBO':
        return EBOWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == 'EBO_B':
        return EBO_BWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == 'KPCABO':
        return KPCABOWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == 'pyCMA':
        return Py_CMA_ES_Wrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget,
                             random_seed=random_seed)

    if optimizer_name == 'RDUCB':
        return RDUCBWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == 'ALEBO':
        return ALEBOWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    if optimizer_name == 'HEBO':
        return HEBOWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    
    # TODO: NEW ADDENDA TO COMPLY WITH KUDELA, STRIPINIS, RAMIUZKAS
    if optimizer_name == "BAxUS":
        return BAxUSWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    
    if optimizer_name== "SMAC":
        return SMACWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    
    if optimizer_name=="REMBO":
        return REMBOWrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)
    

    if optimizer_name=="BO_botorch":
        return BO_Torch_VanillaBO_Wrapper(func=func, dim=ml_dim, ub=ub, lb=lb, total_budget=ml_total_budget, DoE_size=ml_DoE_size,
                             random_seed=random_seed,sample_zero=sample_zero)

if __name__ == "__main__":
    dim = 3
    total_budget = 200
    doe_size = dim
    seed = 4
    sample_zero = True

    try:
        import ioh
    except ModuleNotFoundError as e:
        print("IOH module not found",e.msg)
    
    # Algorithm alternatives:
    algorithm_name = "REMBO"
    #algorithm_name = "turbom"
    f = get_problem(1, dimension=dim, instance=1, problem_class=ioh.ProblemClass.BBOB)

    opt = wrapopt(algorithm_name, 
                  f,
                  dim, 
                  total_budget,
                  doe_size,
                  seed,
                  sample_zero)
    opt.run(matrix_type="normal",kern_inp_type="X",ARD=False)
