
# Import the libraries
import numpy as np
import ioh 
from typing import List, Union, Tuple, Optional, Callable, Dict
from .ModularMetaData import ModularMetaData

'''
This code is defined as some extension to define modular problems from the
BBOB framework.
@Author:
    - Ivan Olarte Rodriguez

'''



class ModularBBOBProblem(ioh.iohcpp.problem.RealSingleObjective):
    """
        This class is defined to perform a Single Objective Optimization based by building blocks of 
        instances of the BBOB functions. 

        Each target corresponds to the arithmetic mean of multiple instances of the BBOB defined on 
        some latent dimensions.
    """

    #def __init__(self, name, n_variables, instance, is_minimization, bounds, constraints, optimum):
    def __init__(self,
                 fid:Union[int,str],
                 instance:int,
                 latent_dimensionality:int,
                 n_repetitions:int)->object:
        
        r"""
        This is the initialiser of the class. This takes the following parameters:
            
            Args:
            -------------
            - fid: The problem identifier from the BBOB (an integer between 1 to 24 or a name)
            - instance: an integer with the defined instance of the problem.
            - latent_dimensionality: The intrinsic dimensionality of the module (positive integer)
            - n_repetitions: The times the latent dimension is repeated.
        """
        
        # Get one of the problems by using the normal IOH calls to instantiate a problem
        if isinstance(fid,str):
            prob_Id:int = ioh.get_problem_id(fid,ioh.ProblemClass.BBOB)
        else:
            prob_Id:int = fid

        # Generate an instance of the intrinsic/latent problem
        self.__intrinsic_problem:ioh.problem.RealSingleObjective = ioh.get_problem(fid=prob_Id,
                                                                    instance=instance,
                                                                    dimension=latent_dimensionality,
                                                                    problem_class=ioh.ProblemClass.BBOB) 
        
        #Extract the constraints and bounds as objects from intrinsic/latent problem
        actual_bounds = self.__intrinsic_problem.bounds
        actual_constraints:ioh.iohcpp.RealConstraintSet = []

        # Extract the optimum information from actual instance
        intrinsic_optimum:ioh.iohcpp.RealSolution = self.__intrinsic_problem.optimum

        actual_optimum = ioh.iohcpp.RealSolution(
                                                    x=np.tile(intrinsic_optimum.x,n_repetitions).ravel(),
                                                    y=intrinsic_optimum.y
                                                
                                                )

        # Check the minimisation type
        if self.__intrinsic_problem.meta_data.optimization_type == ioh.iohcpp.MIN:
            isminimisation = True
        else:
            isminimisation = False

        #Initialize the superclass, being the BBOB class from IOH
        super().__init__(  "Modular_" + self.__intrinsic_problem.meta_data.name,#name 
                            latent_dimensionality*n_repetitions,#n_variables
                            instance, #instance
                            isminimisation,   #is_minimization
                            actual_bounds,  #bounds 
                            actual_constraints, #constraints
                            actual_optimum #optimum
                        )
        
        # Overload the MetaData
        self.__meta_data = ModularMetaData(problem_id=self.__intrinsic_problem.meta_data.problem_id,
                                          instance= self.__intrinsic_problem.meta_data.instance,
                                          name= "Modular_" + self.__intrinsic_problem.meta_data.name,
                                          n_variables=latent_dimensionality*n_repetitions,
                                          optimization_type=self.__intrinsic_problem.meta_data.optimization_type,
                                          latent_dimensionality=latent_dimensionality
                                          )
        
        super().set_id(self.__meta_data.problem_id)

    def evaluate(self, 
                 x:np.ndarray)->float:
        r"""
            This is an overload of the evaluate function from IOH. This will be computed as
            the average of each evaluation of the intrinsic problem.

            Args:
            -------
            - x (`np.ndarray`): An array (preferably NumPy) to compute the target
        """

        # Convert the array to Numpy (to be safe)
        x_mod:np.ndarray = np.array(x)

        # Reshape the array to evaluate the intrinsic function
        x_reshape:np.ndarray = x_mod.reshape((self.meta_data.n_repetitions,self.meta_data.latent_dimensionality))

        current_sum:float = 0.0

        for _,arr in enumerate(x_reshape):
            current_sum += self.__intrinsic_problem(arr)
        
        return current_sum/self.meta_data.n_repetitions


    def create(self, id, iid, dim):
        """
        This is defined as some overload function. However, this will not be used and raises a `NotImplementedError`
        """
        raise NotImplementedError()
    
    def reset(self)->None:
        """
        Overload of reset function from the super class
        """
        # Perform the reset method by following the super class
        super().reset()

        # Reset the intrinsic problem
        self.__intrinsic_problem.reset()

    def attach_logger_to_intrinsic_problem(self, 
                                           logger:ioh.logger.AbstractLogger)->None:
        
        """
        This function appends a separate data logger in case for the intrinsic problem
        """

        self.__intrinsic_problem.attach_logger(logger)
    
    def detach_logger_from_intrinsic_problem(self)->None:
        """
        This function detaches the logger from the intrinsic problem
        """
        return self.__intrinsic_problem.detach_logger()

    
    def detach_logger(self)->None:
        r"""
        This is an overload function from the super class. To be safe, this detaches the logger from this instance and the intrinsic problem instance
        """

        # Use the super_class definition
        super().detach_logger()

        self.detach_logger_from_intrinsic_problem()
    
    # @property
    # def intrinsic_problem(self)->ioh.problem.BBOB:
    #     "This property just returns the intrinsic problem computed by using the BBOB"
    #     return self.__intrinsic_problem
    
    @property
    def meta_data(self)->ModularMetaData:
        "This property explotes the overload of the meta-data"
        return self.__meta_data