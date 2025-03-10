# Import the libraries
import numpy as np
import ioh 
from ioh import wrap_problem
from ioh import ProblemClass
from ioh.iohcpp import MAX,MIN
from typing import List, Union, Tuple, Optional, Callable, Dict
from .ModularMetaData import ModularMetaData
from .ModularLogInfo import ModularLogInfo
from ioh.iohcpp import RealSolution

'''
This code is defined as some extension to define modular problems from the
BBOB framework.
@Author:
    - Ivan Olarte Rodriguez

'''

def bimodal_function(x:np.ndarray)->float:
    r"""
    Evaluate the function x**5 - x**3

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000
    if x.size == 1:
        result = 200*((x[0]/5)**5 -(x[0]/5)**3)
    else:
        result = np.nan

    return result 

def multimodal_function(x:np.ndarray)->float:
    r"""
    Evaluate the function 1 + x**2 - cos(10x)

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000
    if x.size == 1:
        result = 1 + x[0]**2 - np.cos(20*x[0])
    else:
        result = np.nan

    return result 

def plateau_function(x:np.ndarray)->float:
    r"""
    Evaluate the function (1-x)*exp(-x**2)

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000
    if x.size == 1:
        result = (1-x[0])*np.exp(-x[0]**2)
    else:
        result = np.nan

    return result 

def ackley(x:np.ndarray)->float:
    r"""
    Evaluate the Ackley function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000
    if x.size == 1:
        result = -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0])))
    else:
        result = np.nan

    return result

def gramacy_lee(x:np.ndarray)->float:
    r"""
    Evaluate the Gramacy & Lee function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    # Transform the input to fit into -5 to 5
    xmin_new = 0.5
    xmax_new = 2.5

    xmax_old = 5.0
    xmin_old = -5.0

    # Rescale the x
    x = (x-xmin_old)/(xmax_old-xmin_old)*(xmax_new-xmin_new) + xmin_new



    if x.size == 1:
        result = np.sin(10*np.pi*x[0])/(2*x[0]) + (x[0]-1)**4
    else:
        result = np.nan

    return result

def levy(x:np.ndarray)->float:
    r"""
    Evaluate the Levy function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    if x.size == 1:
        w = 1 + (x[0]*2-1)/4
        result = (np.sin(np.pi*w)**2 + 
                  (w-1)**2*(1+10*np.sin(np.pi*w+1)**2) +
                  (w-1)**2*(1+np.sin(2*np.pi*w)**2))
    else:
        result = np.nan

    return result

def rastrigin(x:np.ndarray)->float:
    r"""
    Evaluate the Rastrigin function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    if x.size == 1:
        result = 10*x[0]**2 - 10*np.cos(2*np.pi*x[0])
    else:
        result = np.nan

    return result

def schwefel(x:np.ndarray)->float:
    r"""
    Evaluate the Schwefel function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()*100

    result:float = 2000000

    if x.size == 1:
        result = 418.9829 - x[0]*np.sin(np.sqrt(np.abs(x[0])))
    else:
        result = np.nan

    return result

def sum_different_powers_function(x:np.ndarray)->float:
    r"""
    Evaluate the sum of different powers function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    if x.size == 1:
        result = np.sum(np.abs(x)**(np.arange(1,x.size+1)))
    else:
        result = np.nan

    return result

def forrester(x:np.ndarray)->float:
    r"""
    Evaluate the Forrester function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    # Modify the input to fit into the range 0 to 1 from -5 to 5    
    x = (x+5)/10

    result:float = 2000000

    if x.size == 1:
        result = (6*x[0]-2)**2*np.sin(12*x[0]-4)
    else:
        result = np.nan

    return result

def sphere(x:np.ndarray)->float:
    r"""
    Evaluate the Sphere function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    if x.size == 1:
        result = np.sum(x**2)
    else:
        result = np.nan

    return result

def alpine(x:np.ndarray)->float:
    r"""
    Evaluate the Alpine function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    if x.size == 1:
        norm_x = np.linalg.norm(x,ord=2)  # Compute Euclidean norm (sqrt of sum of squares)
        result =  1 - np.cos(2 * np.pi * norm_x) + 0.1 * norm_x
    else:
        result = np.nan

    return result

def styblinski_tank(x:np.ndarray)->float:
    r"""
    Evaluate the Styblinski-Tank function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    if x.size == 1:
        result = 0.5*(x[0]**4-16*x[0]**2+5*x[0])
    else:
        result = np.nan

    return result

def salomon(x:np.ndarray)->float:
    r"""
    Evaluate the Salomon function

    Args:
    -----------
    - x: The input array whose size shall be equal to 1
    """

    x = np.array(x).ravel()

    result:float = 2000000

    if x.size == 1:
        result = 1-np.cos(2*np.pi*np.sqrt(x[0]**2)) + 0.1*np.sqrt(x[0]**2)
    else:
        result = np.nan

    return result



class ModularNonBBOBProblem(ioh.iohcpp.problem.RealSingleObjective):
    """
        This class is defined to perform a Single Objective Optimization based by building blocks of 
        instances 1D problems. 

        Each target corresponds to the arithmetic mean of multiple instances of the functions defined
        on some 1D problems
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
                          "sphere":sphere,
                          "alpine":alpine,
                          "styblinski_tank":styblinski_tank,
                          "salomon":salomon}
    
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
                             11:"sphere",
                             12:"alpine",
                             13:"styblinski_tank",
                             14:"salomon"}
    
    __optimum_dict:dict = {"bimodal":{'x':5*np.sqrt(3/5),
                                      'y':bimodal_function(5*np.sqrt(3/5))},
                            "multimodal":{'x':0.0,
                                      'y':multimodal_function(0)},
                            "plateau":{'x':0.5+np.sqrt(3/4),
                                       'y':plateau_function(0.5+np.sqrt(3/4))},
                            "ackley":{'x':0.0, 'y':ackley(0.0)},
                            "gramacy_lee":{'x':-4.757183, 'y':gramacy_lee(-4.757183)},
                            "levy":{'x':0.5, 'y':levy(0.5)},
                            "rastrigin":{'x':0.0, 'y':rastrigin(0.0)},
                            "schwefel":{'x':420.9687/100, 'y':schwefel(420.9687/100)},
                            "sum_different_powers":{'x':0, 'y':sum_different_powers_function(0.0)},
                            "forrester":{'x':0.7572*10-5, 'y':forrester(0.7572*10-5)},
                            "sphere":{'x':0.0, 'y':sphere(0.0)},
                            "alpine":{'x':0.0, 'y':alpine(0.0)},
                            "styblinski_tank":{'x':-2.903534, 'y':styblinski_tank(-2.903534)},
                            "salomon":{'x':0.0, 'y':salomon(0.0)}}
    
    __modes:dict = {"average":1,"maximum":2}
    
    __latent_dimensionality:int = 1
    __instance:int = 0
    
    #def __init__(self, name, n_variables, instance, is_minimization, bounds, constraints, optimum):
    def __init__(self,
                 fid:Union[int,str],
                 n_repetitions:int,
                 mode:Optional[Union[str,int]]="average")->object:
        
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
        
        # Set the mode
        self.mode = mode
        
        
        # Extract the optimum information from actual instance
        intrinsic_optimum = RealSolution(x=[self.__optimum_dict[func_name]['x']],
                                                      y=self.__optimum_dict[func_name]['y'])
        
        # Define a wrapper 
        self.__intrinsic_problem:ioh.problem.RealSingleObjective = wrap_problem(function=func,
                                                                                name=func_name,
                                                                                dimension=self.__latent_dimensionality,
                                                                                instance=self.__instance,
                                                                                optimization_type=MIN,
                                                                                lb=-5,
                                                                                ub=5
        )
        
        #Extract the constraints and bounds as objects from intrinsic/latent problem
        actual_bounds = self.__intrinsic_problem.bounds
        actual_constraints:ioh.iohcpp.RealConstraintSet = []

       


        
        actual_optimum = RealSolution(
            x=np.tile(intrinsic_optimum.x,n_repetitions).ravel(),
            y=intrinsic_optimum.y)

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

        # This is to set in case the mode is set as the average
        if self.mode == "average":
            current_sum:float = 0.0

            for _,arr in enumerate(x_reshape):
                current_sum += self.__intrinsic_problem(arr)
            
            return current_sum/self.meta_data.n_repetitions

        # This is to set in case the mode is set as the maximum
        elif self.mode == "maximum":
            current_max:float = -np.inf

            for _,arr in enumerate(x_reshape):
                current_max = max(current_max,self.__intrinsic_problem(arr))
            
            return current_max


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
    def mode(self)->str:
        "This property returns the mode of the problem"
        return self.__mode
    
    @mode.setter
    def mode(self, mode:Union[str,int])->None:
        "This property sets the mode of the problem"

        if isinstance(mode,str):
            if mode in [*self.__modes]:
                self.__mode = mode
            else:
                raise AttributeError(f"The mode given is a string that does not match the options in {[*self.__modes]}",
                                     mode,"mode")
        
        elif isinstance(mode,int):
            if mode in [*self.__modes.values()]:
                self.__mode = mode
            else:
                raise AttributeError(f"The mode given is an integer that does not match the options in {[*self.__modes.values()]}",
                                     mode,"mode")
        
        else:
            raise AttributeError(f"The mode given is not an integer or a string",
                                 mode,"mode")


