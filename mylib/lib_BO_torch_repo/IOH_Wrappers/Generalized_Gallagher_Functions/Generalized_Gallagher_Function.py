import numpy as np
from typing import (List ,
                    Union ,
                    Optional)
from ioh.iohcpp.problem import RealSingleObjective
from ioh.iohcpp import RealBounds, RealSolution


def _fpen(x:np.ndarray)->float:
    r"""
    Compute the value of the fpen function at x by using the BBOB-COCO definition.

    Args:
    ---------------

    - x: Input vector of shape (dim,) within the range (-5,5).
    """
    # Reshape the input vector
    x = x.ravel()
    dim = x.size

    # Compute the function value
    accumulator:int = 0

    for i in range(dim):
        accumulator += max(0,np.abs(x[i])-5)**2

    return accumulator

def _fopt(x:np.ndarray,
         xopt:np.ndarray,
         norm_type:Union[str,int]=2)->float:
    
    r"""
    This is a function that generates a penalty based on the norm (or distance)
    from the optimal point.
    """
    dim:int = xopt.size

    if isinstance(norm_type,int):
        #return np.linalg.norm(x-xopt,ord=norm_type)/(10*dim**(1/norm_type))
        return np.linalg.norm(x-xopt,ord=norm_type)
    else:
        return np.linalg.norm(x-xopt,ord=norm_type)

class GeneralizedGallagherFunction(RealSingleObjective):

    r"""
        This is an extension of the Gallagher function by taking the developments from the BBOB-COCO Testbench
        (see: https://coco-platform.org/testsuites/bbob/functions/f21.html) as well as the original paper
        from Gallagher & Yuan ("A General-Purpose Tunable Landscape Generator", 
        IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 10, NO. 5, OCTOBER 2006).

        To follow the other benchmarks, this class extends the IOH framework:
        see: https://iohprofiler.github.io/IOHexperimenter/
    """
    
    def __init__(self,
                 num_peaks:Optional[int]=21, 
                 dim:Optional[int]=2, 
                 instance:Optional[int] =1,
                 norm_type:Optional[Union[str,int]]=2)->None:
        
        r"""
        The main initializer of the class.

        Args:
        -------------
      
        - x: Input vector of shape (dim,).
        - num_peaks: `int`: Number of Gaussian peaks.
        - dim: `int`: Dimensionality of the input space.
        - instance: `Optional[int]`; an instance of np.random.RandomState for reproducibility.
        - norm_type: `Optional[Union[str,int]]`: The norm to compute a penalty for not being close to the optimum value
    """
        
        # Input checks
        if not (num_peaks >=1 and isinstance(num_peaks,int)):
            raise ValueError("The number of peaks should be greater than 1")
        
        if not (dim >=1 and isinstance(dim,int)):
            raise ValueError("The dimension of the problem should be a positive integer")
        
        #Instantiate a random state given the instance
        random_state = np.random.RandomState(instance)

        # Store the norm type
        self._norm_type:Union[int,str] = norm_type

        # Generate random peak properties
        self._peak_locations:np.ndarray = np.vstack(
                                                        ( 
                                                            random_state.uniform(-3.99, 3.99, (1, dim)),
                                                            random_state.uniform(-4.99, 4.99, (num_peaks-1, dim)) 
                                                        ) 
                                                    )
        
        self._peak_heights:List[float] = []


    
        self._q_list:List[np.ndarray] = []
        self._r_list:List[np.matrix] = []


        # Compute the covariance matrices and compute the spectral eigendecomposition
        for i in range(num_peaks):

            if i == 0:
                self._peak_heights.append(10)
            else:
                self._peak_heights.append(1.1+8*(i)/(num_peaks-1))

            # Generate random matrices
            cov = random_state.uniform(-1,1,(dim, dim))

            cov = np.dot(cov.T,cov)

            # Perform the computation 
            q, r = np.linalg.eigh(cov)
            self._q_list.append(q)
            self._r_list.append(r)
        
        name = "Generalized Gallagher Function with {0} peaks".format(num_peaks)
        bounds = RealBounds(dim,-5,5)
        optimum = RealSolution(
                                    x=self._peak_locations[0].tolist(),
                                    y=self._compute_value(self._peak_locations[0])
                              )
        
        # Initialize the superclass
        super().__init__(
                            name=name,
                            n_variables=dim,
                            instance=instance,
                            is_minimization=True,
                            bounds=bounds,
                            constraints=[],
                            optimum= optimum
                        )
    
    def _compute_value(self,x:Union[np.ndarray,List[float]])->float:
        r"""
        This function computes the value of the function (without the penalties)

        Args:
        -----------------
        - x: `Union[np.ndarray,List[float]]`: The vector denoting some point to compute the function.
        """

        # Compute the dimensionality (again!)
        dim = self._peak_locations.shape[1]

        if isinstance(x,list):
            x = np.asarray(x)
        
        # Reshape the array
        x = x.ravel()

        # Compute the function value
        values = []

        
        for i in range(len(self._peak_locations)):
            diff = (x - self._peak_locations[i]).reshape(-1,1)
            exponent = (-(1/(2*dim)) * (diff.T @ ( self._r_list[i] @ np.diag(np.divide(1,self._q_list[i]) ) @  self._r_list[i].T )   @ diff)).ravel()[0]
            value = self._peak_heights[i] * np.exp(exponent)
            values.append(value)

        return (10-np.max(values))**2
    
    def evaluate(self,x:Union[np.ndarray,List[float]])->float:
        r"""
        This is an overload of the evaluate function to compute the raw function
        """

        # Compute the dimensionality (again!)
        dim = self._peak_locations.shape[1]

        if isinstance(x,list):
            x = np.asarray(x)
        
        # Reshape the array
        x = x.ravel()

        return self._compute_value(x) + _fopt(x=x,xopt=self._peak_locations[0],norm_type=self._norm_type) + _fpen(x)
    
    
    def __del__(self, *args, **kwargs):
        self.detach_logger()


    def create(self, id, iid, dim):
        raise NotImplementedError
    

        
        
# Example usage:
#x = #np.array([3.73,0.376])
# x = np.array([-5,4.1,3,2,1]*10)
# funcc = GeneralizedGallagherFunction(num_peaks=101, 
#                                      dim=50, 
#                                      instance=5,
#                                      norm_type=2)
# value = funcc(x)

# print("Function value at x:", value)
# print(f"Function value at optimum: x={funcc.optimum.x}, ", funcc.optimum.y)
