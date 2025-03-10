from ioh.iohcpp import LogInfo
from ioh.iohcpp.logger.property import AbstractProperty
from typing import List, ClassVar, Tuple, Dict



### ----------------------------------------------------
### CLASS DEFINITION
### ----------------------------------------------------


class Regret(AbstractProperty):

    def name(self)->str:
        return "Regret"
    
    def call_to_string(self, arg0, arg1):
        return super().call_to_string(arg0, arg1)
    
    def __call__(self, arg0):
        return super().__call__(arg0)
    

