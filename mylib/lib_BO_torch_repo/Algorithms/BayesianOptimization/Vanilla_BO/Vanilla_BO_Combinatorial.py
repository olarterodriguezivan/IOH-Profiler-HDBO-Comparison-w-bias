from .Vanilla_BO import Vanilla_BO
from IOH_Wrappers.ModularBBOBProblem import ModularBBOBProblem
from IOH_Wrappers.ModularNonBBOBProblem import ModularNonBBOBProblem
from typing import Union, Callable, Optional, List, Tuple
from ioh.iohcpp.problem import RealSingleObjective, BBOB
from pyDOE import lhs
from functools import partial
import numpy as np
import torch
import os
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputStandardize, Normalize
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

#from gpytorch.mlls.marginal_log

ALLOWED_ACQUISITION_FUNCTION_STRINGS:tuple = ("expected_improvement",
                                              "probability_of_improvement",
                                              "upper_confidence_bound")

ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS:dict = {"EI":"expected_improvement",
                                                       "PI":"probability_of_improvement",
                                                       "UCB":"upper_confidence_bound"}


class Vanilla_BO_Combinatorial(Vanilla_BO):
    """
    This is a handler class to perform the Vanilla BO with combinatorial repetitions of the target

    """
    def __init__(self, budget, n_DoE=0, random_seed=43,acquisition_function:str="expected_improvement",**kwargs):

        # Call the superclass
        super().__init__(budget, 
                         n_DoE,
                         acquisition_function,
                         random_seed,
                           **kwargs)
        
        # Initialise the intrinsic dimension
        self.intrinsic_dimension:Union[int,None] = None

        

        
    def __str__(self):
        return "This is an instance of Vanilla BO Optimizer"

    def __call__(self, problem:Union[RealSingleObjective,BBOB,ModularBBOBProblem,ModularNonBBOBProblem,Callable],
                 dim:Optional[int]=-1, 
                 bounds:Optional[np.ndarray]=None,
                 intrinsic_dimension:Optional[int] = None,
                 **kwargs)-> None:

        # Call the superclass (namely the AbstractBayesianOptimizer) to run the initial sampling of the problem
        super(Vanilla_BO,self).__call__(problem, dim, bounds, **kwargs)

        if isinstance(problem,(ModularBBOBProblem,ModularNonBBOBProblem)):
            self.intrinsic_dimension = problem.meta_data.latent_dimensionality
        else:
            # In this case the intrinsic dimension should be given
            if isinstance(intrinsic_dimension,int) and np.remainder(self.dimension,intrinsic_dimension) == 0:
                self.intrinsic_dimension = intrinsic_dimension
            else:
                raise AttributeError("The intrinsic dimension cannot be assigned as this is not a multiple of the global problem dimension",
                                     name="intrinsic dimension",
                                     obj=intrinsic_dimension) 


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
        
        # Print Message
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
        
        import math
        import itertools

        # Convert bounds array to Torch
        bounds_torch:Tensor = torch.from_numpy(self.bounds.transpose()).detach()

        # Convert the arrays first to Numpy
        train_x:np.ndarray = np.array(self.x_evals).reshape((-1,self.dimension)) 
        train_obj:np.ndarray = np.array(self.f_evals).reshape((-1,1))

        # Arrange Space to augment the arrays
        train_x_aug:List[np.ndarray] = []
        train_obj_aug:List[float] = []

        main_single_array:np.ndarray = np.arange(int(self.dimension/self.intrinsic_dimension))

        

        for idx,cur_arr in enumerate(train_x):

            # reshape current iterated array
            main_struct_arr:np.ndarray = cur_arr.reshape((-1,
                                                          self.intrinsic_dimension))
            
            cur_f_eval = train_obj.ravel()[idx]

            # Now start with a permutation approach
            permutations_array:list = itertools.permutations(main_single_array.ravel().tolist())
            for cur_per in permutations_array:
                modified_structure:np.ndarray = main_struct_arr[cur_per,:]
                evaluation_array:np.ndarray = modified_structure.ravel(order="C")

                # Store in augmented list
                train_x_aug.append(evaluation_array)

                train_obj_aug.append(cur_f_eval)
        
        # Now convert the augmented arrays to Torch
        train_x_aug_torch:Tensor = torch.from_numpy(np.array(train_x_aug).reshape((-1,self.dimension))).detach()
        train_obj_aug_torch:Tensor = torch.from_numpy(np.array(train_obj_aug).reshape((-1,1))).detach()

        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-07))


        self.__model_obj:SingleTaskGP =  SingleTaskGP(train_x_aug_torch,
                                                      train_obj_aug_torch,
                                                      #train_Yvar,
                                                      likelihood=likelihood,
                                                      outcome_transform=Standardize(m=1),
                                                      input_transform=Normalize(d=train_x.shape[-1],
                                                                                bounds=bounds_torch)
        )
    
    def optimize_acqf_and_get_observation(self)->Tensor:
        """Optimizes the acquisition function, and returns a new candidate."""
        # Recycle the super_class 
        new_x = super().optimize_acqf_and_get_observation()

        return new_x


    def __repr__(self):
        return super().__repr__()
    
    def reset(self):
        # call the Super Class
        super().reset()

        # Assign the intrinsic dimension to None
        self.intrinsic_dimension = None
    
    @property
    def intrinsic_dimension(self)->Union[int,None]:
        return self.__intrinsic_dimension
    
    @intrinsic_dimension.setter
    def intrinsic_dimension(self, new_intrinsic_dimension:Union[int,None]):

        if isinstance(new_intrinsic_dimension,int) and new_intrinsic_dimension >= 1:
            # Set the intrinsic dimension in this conditions
            self.__intrinsic_dimension = new_intrinsic_dimension
        elif new_intrinsic_dimension is None:
            self.__intrinsic_dimension = new_intrinsic_dimension
        else:
            raise AttributeError("Cannot assign the intrinsic dimension to the given value",
                                 name="intrinsic_dimension",
                                 obj= new_intrinsic_dimension)
    
    



