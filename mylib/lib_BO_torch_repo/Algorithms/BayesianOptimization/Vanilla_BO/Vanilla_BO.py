from ..AbstractBayesianOptimizer import AbstractBayesianOptimizer
from typing import Union, Callable, Optional
from ioh.iohcpp.problem import RealSingleObjective, BBOB
from pyDOE import lhs
from functools import partial
import numpy as np
import torch
import os
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.acquisition.analytic import (ExpectedImprovement,
                                          ProbabilityOfImprovement, 
                                          UpperConfidenceBound, 
                                          AnalyticAcquisitionFunction)
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import GaussianLikelihood

#from gpytorch.mlls.marginal_log

ALLOWED_ACQUISITION_FUNCTION_STRINGS:tuple = ("expected_improvement",
                                              "probability_of_improvement",
                                              "upper_confidence_bound")

ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS:dict = {"EI":"expected_improvement",
                                                       "PI":"probability_of_improvement",
                                                       "UCB":"upper_confidence_bound"}


class Vanilla_BO(AbstractBayesianOptimizer):
    def __init__(self, budget, n_DoE=0, acquisition_function:str="expected_improvement",
                 random_seed:int=43,**kwargs):

        # Call the superclass
        super().__init__(budget=budget, 
                         n_DoE=n_DoE,
                          random_seed= random_seed,
                            **kwargs)

        # Check the defaults
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        dtype = torch.double
        smoke_test = os.environ.get("SMOKE_TEST")

        # Set up the main configuration
        self.__torch_config:dict = {"device":device,
                                    "dtype":dtype,
                                    "SMOKE_TEST":smoke_test,
                                    "BATCH_SIZE":3 if not smoke_test else 2,
                                    "NUM_RESTARTS": 10 if not smoke_test else 2,
                                    "RAW_SAMPLES": 512 if not smoke_test else 32}
        
        # Set-up the acquisition function
        self.__acq_func:Optional[Union[AnalyticAcquisitionFunction,Callable]] = None
        self.acquistion_function_name = acquisition_function

        
    def __str__(self):
        return "This is an instance of Vanilla BO Optimizer"

    def __call__(self, problem:Union[RealSingleObjective,Callable], 
                 dim:Optional[int]=-1, 
                 bounds:Optional[np.ndarray]=None, 
                 **kwargs)-> None:

        # Call the superclass to run the initial sampling of the problem
        super().__call__(problem, dim, bounds, **kwargs)

        # Get a default beta (for UCB)
        beta = kwargs.pop("beta",0.2)

        # Run the model initialisation
        self._initialise_model(**kwargs)

        # Start the optimisation loop
        for cur_iteration in range(self.budget-self.n_DoE):

            # Set up the acquistion function
            self.acquisition_function = self.acquisition_function_class(
                model= self.__model_obj,
                best_f= self.current_best,
                maximize= self.maximisation,
                beta=beta,
                kwargs=kwargs
            )
            
            new_x = self.optimize_acqf_and_get_observation()

            # Append the new values
            for _, new_x_arr in enumerate(new_x):
                new_x_arr_numpy:np.ndarray = new_x_arr.detach().numpy().ravel()

                # Append the new value
                self.x_evals.append(new_x_arr_numpy)

                # Evaluate the function
                new_f_eval:float = problem(new_x_arr_numpy)

                # Append the function evaluation
                self.f_evals.append(new_f_eval)

                # Increment the number of function evaluations
                self.number_of_function_evaluations +=1
                
            # Assign the new best
            self.assign_new_best()

            # Print best to screen if verbose
            if self.verbose:
                print(f"Current Iteration:{cur_iteration+1}",
                       f"Current Best: x:{self.x_evals[self.current_best_index]} y:{self.current_best}"
                  ,flush=True)
            

            # Re-fit the GPR
            self._initialise_model()
        
        print("Optimisation Process finalised!")

    def assign_new_best(self):
        # Call the super class
        super().assign_new_best()

    
    def _initialise_model(self, **kwargs):
        r"""
        This function initialise/fits the Gaussian Process Regression
        

        Args:
        -------
        - **kwargs: Left these keyword arguments for upcoming developments
        """

        # Convert bounds array to Torch
        #bounds_torch:Tensor = torch.from_numpy(self.bounds.transpose()).detach()

        # Convert the initial values to Torch Tensors
        train_x:np.ndarray = np.array(self.x_evals).reshape((-1,self.dimension)) 
        train_x:Tensor = torch.from_numpy(train_x).detach()

        train_obj:np.ndarray = np.array(self.f_evals).reshape((-1,1))
        train_obj:Tensor = torch.from_numpy(train_obj).detach()

        self.__model_obj:SingleTaskGP =  SingleTaskGP(train_x,
                                                      train_obj,
                                                      #train_Yvar,
                                                      #likelihood=GaussianLikelihood(noise_prior=1e-06),
                                                      outcome_transform=Standardize(m=1,min_stdv=5e-06),
                                                      #input_transform=Normalize(d=train_x.shape[-1],
                                                      #                          bounds=bounds_torch)
        )
    
    def optimize_acqf_and_get_observation(self)->Tensor:
        """Optimizes the acquisition function, and returns a new candidate."""
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=self.acquisition_function,
            bounds=torch.from_numpy(self.bounds.transpose()).detach(),
            q=1,#self.__torch_config['BATCH_SIZE'],
            num_restarts=self.__torch_config['NUM_RESTARTS'],
            raw_samples=self.__torch_config['RAW_SAMPLES'],  # used for intialization heuristic
            options={"batch_limit": 20, "maxiter": 2000},
            sequential=True,
            return_best_only=True
        )
        

        # observe new value
        new_x = candidates.detach()

        new_x = new_x.reshape(shape=((1,-1))).detach()

 

        return new_x


    def __repr__(self):
        return super().__repr__()
    
    def reset(self):
        return super().reset()
    
    @property
    def torch_config(self)->dict:
        return self.__torch_config
    
    @property
    def acquistion_function_name(self)->str:
        return self.__acquisition_function_name
    
    @acquistion_function_name.setter
    def acquistion_function_name(self, new_name:str)->None:
        
        # Remove some spaces
        new_name = new_name.strip()
        
        # Start with a dummy variable
        dummy_var:str = ""

        # Check in the reduced
        if new_name in [*ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS]:
            # Assign the name
            dummy_var = ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS[new_name]
        else:
            if new_name.lower() in ALLOWED_ACQUISITION_FUNCTION_STRINGS:
                dummy_var = new_name
            else:
                raise ValueError("Oddly defined name")
        
        self.__acquisition_function_name = dummy_var
        # Run to set up the acquisition function subclass
        self.set_acquisition_function_subclass()
    
    def set_acquisition_function_subclass(self)->None:

        if self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[0]:
            self.__acq_func_class = ExpectedImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[1]:
            self.__acq_func_class = ProbabilityOfImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[2]:
            self.__acq_func_class = UpperConfidenceBound
    
    @property
    def acquisition_function_class(self)->Callable:
        return self.__acq_func_class
    
    @property
    def acquisition_function(self)->AnalyticAcquisitionFunction:
        """
        This returns the stored defined acquisition function defined at some point 
        of the loop
        """
        return self.__acq_func
    
    @acquisition_function.setter
    def acquisition_function(self, new_acquisition_function:AnalyticAcquisitionFunction)->None:
        r"""
        This is the setter function to the new acquisition function
        """

        if issubclass(type(new_acquisition_function),AnalyticAcquisitionFunction):
            # Assign in this case
            self.__acq_func = new_acquisition_function
        else:
            raise AttributeError("Cannot assign the acquisition function as this does not inherit from the class `AnalyticAcquisitionFunction` ",
                                 name="acquisition_function",
                                 obj=self.__acq_func)


