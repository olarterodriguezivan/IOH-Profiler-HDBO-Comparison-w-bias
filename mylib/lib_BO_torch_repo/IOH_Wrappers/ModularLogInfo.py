from ioh.iohcpp import LogInfo
from ioh.iohcpp import RealSolution


class ModularLogInfo(LogInfo):
    def __init__(self,
        evaluations: int,
        raw_y_best: float,
        transformed_y: float,
        transformed_y_best: float,
        current: RealSolution,
        optimum: RealSolution):
        # Add any additional initialization here

        # Call the parent class's __init__ method
        super().__init__(evaluations, 
                         raw_y_best, 
                         transformed_y, 
                         transformed_y_best, 
                         current, 
                         optimum)

    # Add any additional methods or overwrite existing methods as needed