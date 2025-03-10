# Import the libraries
import numpy as np
import ioh 
from ioh import wrap_problem
from ioh import ProblemClass
from ioh.iohcpp import MAX,MIN
from typing import List, Union, Tuple, Optional, Callable, Dict
from .ModularMetaData import ModularMetaData
from ioh.iohcpp import RealSolution
from .ModularNonBBOBProblem import (bimodal_function,
                                    multimodal_function,
                                    plateau_function,
                                    ackley,
                                    gramacy_lee,
                                    levy,
                                    rastrigin,
                                    schwefel,
                                    sum_different_powers_function,
                                    forrester,
                                    sphere)

'''
This code is defined as some extension to define modular problems from the
BBOB framework.
@Author:
    - Ivan Olarte Rodriguez

'''


class ModularNonBBOBProblemWithPerturbedDistance(ioh.iohcpp.problem.RealSingleObjective):
    """
        This class is defined to perform a Single Objective Optimization based by building blocks of 
        instances 1D problems. 

        In this case, the problem is defined as a linear combination of each of the parameters.
    """

    __namespace_dict:dict = {"bimodal":bimodal_function,
                          "multimodal":multimodal_function,
                          "plateau":plateau_function,
                          "ackley":ackley,
                          "gramacy_lee":gramacy_lee,
                          "levy":levy,
                          "rastrigin":rastrigin,
                          "schwefel":schwefel,
                          "sum_different_powers":sum_different_powers_function,
                          "forrester":forrester,
                          "sphere":sphere}
    
    __idx_space_dict:dict = {1:"bimodal",
                             2:"multimodal",
                             3:"plateau",
                             4:"ackley",
                             5:"gramacy_lee",
                             6:"levy",
                             7:"rastrigin",
                             8:"schwefel",
                             9:"sum_different_powers",
                             10:"forrester",
                             11:"sphere"}
    
    __optimum_dict:dict = {"bimodal":{'x':5*np.sqrt(3/5),
                                      'y':bimodal_function(5*np.sqrt(3/5))},
                            "multimodal":{'x':0.0,
                                      'y':multimodal_function(0)},
                            "plateau":{'x':0.5+np.sqrt(3/4),
                                       'y':plateau_function(0.5+np.sqrt(3/4))},
                            "ackley":{'x':0.0, 'y':ackley(0.0)},
                            "gramacy_lee":{'x':-4.757183, 'y':gramacy_lee(-4.757183)},
                            "levy":{'x':1.0, 'y':levy(1.0)},
                            "rastrigin":{'x':0.0, 'y':rastrigin(0.0)},
                            "schwefel":{'x':420.9687/100, 'y':schwefel(420.9687/100)},
                            "sum_different_powers":{'x':0, 'y':sum_different_powers_function(0.0)},
                            "forrester":{'x':0.7572*10-5, 'y':forrester(0.7572*10-5)},
                            "sphere":{'x':0.0, 'y':sphere(0.0)}}
    
    __modes:dict = {"average":1,"maximum":2}
    
    __latent_dimensionality:int = 1
    
    #def __init__(self, name, n_variables, instance, is_minimization, bounds, constraints, optimum):
    def __init__(self,
                 fid:Union[int,str],
                 n_repetitions:int,
                 norm_type:Union[str,int]=2,
                 instance:int=1,
                 )->object:
        
        r"""
        This is the initialiser of the class. This takes the following parameters:
            
            Args:
            -------------
            - fid: The problem identifier "either a name or an identifier between 1 to 10"
            - n_repetitions: The times the latent dimension is repeated.
        """
        
        # Start the function and name as some empty variables
        func:Callable = None
        func_name:str = None
        func_id:int = 0

        # Get one of the problems
        if isinstance(fid,str):
            # Check if the string matches one of the names
            if fid in [*self.__namespace_dict]:
                func = self.__namespace_dict[fid]
                func_name = fid

                # Do the inverse mapping of the dictionary
                inv_map:dict = {v: k for k, v in self.__idx_space_dict.items()}
                # Get the id
                func_id = inv_map[fid]
            else:
                raise AttributeError(f"The fid given is a string that does not match the options in {[*self.__namespace_dict]}",
                                     fid,"fid")
        elif isinstance(fid,int):

            # Check if the integer is included in the set
            if fid in [*self.__idx_space_dict]:
                # Set the id
                func_id = fid
                # Get the value
                func_name = self.__idx_space_dict[fid]
                func = self.__namespace_dict[func_name]
            
            else:
                raise AttributeError(f"The fid given is not an integer included in the set: {[*self.__idx_space_dict]}",
                                     "fid",fid)
        else:
            raise AttributeError(f"The fid given is not an integer or a string",
                                 "fid",fid)

        # Define the norm type and the instance
        self.norm_type = norm_type
        self.__instance = instance if instance > 0 else None # By doing this, an error will be generated if the instance is not valid
        
        
        # Extract the optimum information from actual instance
        intrinsic_optimum = RealSolution(x=[self.__optimum_dict[func_name]['x']],
                                                      y=self.__optimum_dict[func_name]['y'])
        
        # Instantiate this optimum as part of the class
        self.__intrinsic_optimum = intrinsic_optimum
        
        
        # Define a wrapper 
        self.__intrinsic_problem:ioh.problem.RealSingleObjective = wrap_problem(function=func,
                                                                                name=func_name,
                                                                                dimension=self.__latent_dimensionality,
                                                                                instance=self.__instance,
                                                                                optimization_type=MIN,
                                                                                lb=-5,
                                                                                ub=5
        )

        # Set the id of the intrinsic problem as the same as the function id in this domain
        self.__intrinsic_problem.set_id(func_id)

        # Get the origin (or one instance of the minimum)
        self.__origin_point:np.ndarray = self._set_origins(n_repetitions)

        # Compute the maximum distance to be obtained given the bounds
        self.__max_distance:float = self._get_max_distance(self.__origin_point,self.norm_type)


        #Extract the constraints and bounds as objects from intrinsic/latent problem
        actual_bounds = self.__intrinsic_problem.bounds
        actual_constraints:ioh.iohcpp.RealConstraintSet = []

        
        actual_optimum = RealSolution(
            x=self.__origin_point.ravel(),
            y=intrinsic_optimum.y
            )

        # Check the minimisation type
        if self.__intrinsic_problem.meta_data.optimization_type == ioh.iohcpp.MIN:
            isminimisation = True
        else:
            isminimisation = False

        #Initialize the superclass, being the BBOB class from IOH
        super().__init__(  "Modular_Non_BBOB_" + self.__intrinsic_problem.meta_data.name,#name 
                            self.__latent_dimensionality*n_repetitions,#n_variables
                            self.__instance, #instance
                            isminimisation,   #is_minimization
                            actual_bounds,  #bounds 
                            actual_constraints, #constraints
                            actual_optimum #optimum
                        )
        
        # Set the function id
        super().set_id(func_id)
        
        # Overload the MetaData
        self.__meta_data = ModularMetaData(problem_id=self.__intrinsic_problem.meta_data.problem_id,
                                          instance= self.__intrinsic_problem.meta_data.instance,
                                          name= "Modular_Non_BBOB_" + self.__intrinsic_problem.meta_data.name,
                                          n_variables=self.__latent_dimensionality*n_repetitions,
                                          optimization_type=self.__intrinsic_problem.meta_data.optimization_type,
                                          latent_dimensionality=self.__latent_dimensionality
                                          ) 
        
    def _set_origins(self, n_repetitions)->np.ndarray:
        r"""
        This function is defined to set the origins "or the minimum" of the n-dimensional problem
        """

        # Set the instance as some random seed
        ### TODO: Check if this is the correct way to set the seed
        np.random.seed(self.instance)

        # Generate a random array
        x = np.random.uniform(low=-5,high=5,size=(n_repetitions,))

        # Sort the array
        x = np.sort(x)

        # Return the array
        return x
    
    @staticmethod
    def _get_max_distance(reference_point:np.ndarray, norm_type:Union[str,int])->float:
        r"""
        This function is defined to compute the maximum possible distance within the bounds of the problem.
        Note the bounds correspond to a rectangle in the n-dimensional space of size (-5,5)*dimensionality

        Args:
        --------
        - reference_point (`np.ndarray`): The reference point to compute the distance
        - norm_type (`Optional[Union[str,int]]`): The norm type to compute the distance. By default, this is the Euclidean norm.
        """

        import itertools

        # NOTE: This is a brute force implementation as this will compute a stupid cartesian product
        lb:float = -5
        ub:float = 5

        n_repetitions:int = reference_point.size

        product_:list = [x for x in itertools.product([lb,ub],repeat=n_repetitions)]

        # Now discard the nodes of the cartesian product which are not sorted
        product_ = [x for x in product_ if all(y <= z for y, z in itertools.pairwise(x))]

        # Now convert the list to a numpy array
        product_:np.ndarray = np.array(product_,dtype=np.float64,subok=False)

        max_dist:float = -np.inf

        # loop through the product
        for _,x_ in enumerate(product_):
            actual_dist:float = np.linalg.norm(x_.ravel()-reference_point.ravel(),ord=norm_type)

            if actual_dist > max_dist:
                max_dist = actual_dist


        # Return the maximum distance
        return max_dist
    
    def _transform_range(self, dist_val:float)->float:
        r"""
        This function is defined to transform the range of the input array to the range of the intrinsic problem
        """

        # Get the optimum of the intrinsic problem
        intrinsic_optimum:RealSolution = self.__intrinsic_optimum
        x_opt = intrinsic_optimum.x

        init_range:tuple = (0,self.__max_distance)
        final_range:tuple = (x_opt,5.0)

        func = (dist_val-init_range[0])/(init_range[1]-init_range[0])*(final_range[1]-final_range[0])+final_range[0]

        return func
        

    def evaluate(self, 
                 x:Union[np.ndarray,List[float]])->float:
        r"""
            This is an overload of the evaluate function from IOH. This will be computed as
            the average of each evaluation of the intrinsic problem.

            Args:
            -------
            - x (`np.ndarray`): An array (preferably NumPy) to compute the target
        """

        # Convert the array to Numpy (to be safe)
        x_mod:np.ndarray = np.array(x, subok=False, copy=True)

        # Flatten the array and sort
        x_sorted:np.ndarray = np.sort(x_mod.ravel())

        # Compute the distance to the origin point
        dist_from_origin:float = np.linalg.norm(x_sorted-self.__origin_point.ravel(),ord=self.norm_type)

        # apply the transformation
        x_transformed:np.ndarray = self._transform_range(dist_from_origin)

        # Compute the intrinsic problem
        intrinsic_eval:float = self.__intrinsic_problem(x_transformed)

        return intrinsic_eval

        #
        #raise NotImplementedError("This function is not implemented yet")


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
    
    @property
    def norm_type(self)->Union[str,int]:
        "This property returns the norm type"
        return self.__norm_type
    
    @norm_type.setter
    def norm_type(self, value:Union[str,int])->None:
        "This property sets the norm type"
        if isinstance(value,str):
            if value in ["average","maximum"]:
                self.__norm_type = self.__modes[value]
            else:
                raise AttributeError(f"The norm type given is not included in the set of modes: {['fro','nuc']}",
                                     value,"norm_type")
        elif isinstance(value,int):
            if value in [1,2]:
                self.__norm_type = value
            else:
                raise AttributeError(f"The norm type given is not included in the set of modes: {[1,2]}",
                                     value,"norm_type")
        else:
            raise AttributeError(f"The norm type given is not a valid type",
                                 value,"norm_type")
    
    @property
    def instance(self)->int:
        "This property returns the instance"
        return self.__instance
    
    
    