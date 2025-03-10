from ioh.iohcpp import MetaData
from ioh.iohcpp import OptimizationType

'''
This code is defined as some extension to define modular problems from the
BBOB framework.
@Author:
    - Ivan Olarte Rodriguez

'''

class ModularMetaData(MetaData):
    r"""
    This is an overload of the problem meta-data to fit with the requirements from modularity
    """

    def __init__(self, 
                 problem_id:int, 
                 instance:int, 
                 name:str, 
                 n_variables:int, 
                 optimization_type:OptimizationType,
                 latent_dimensionality:int):
        
        # Initialise the parent class
        super().__init__(problem_id, instance, name, n_variables, optimization_type)

        #Instantiate the latent dimensionality
        self.__latent_dimensionality:int = latent_dimensionality
    
    @property
    def latent_dimensionality(self)->int:
        """
        Return the latent dimensionality of the modular problem
        """
        return self.__latent_dimensionality
    
    @property 
    def n_repetitions(self)->int:
        """
        Return the number of repetitions of the "latent dimensionality"
        """
        return int(self.n_variables/self.__latent_dimensionality)