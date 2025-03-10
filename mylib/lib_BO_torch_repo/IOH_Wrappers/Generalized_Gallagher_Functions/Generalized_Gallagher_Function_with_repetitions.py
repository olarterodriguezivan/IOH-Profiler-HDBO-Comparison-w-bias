import numpy as np
from typing import (List ,
                    Union ,
                    Optional)
from ioh.iohcpp.problem import RealSingleObjective
from ioh.iohcpp import RealBounds, RealSolution
from itertools import product, permutations


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
         n_repetitions:int,
         norm_type:Union[str,int]=2)->float:
    
    r"""
    This is a function that generates a penalty based on the norm (or distance)
    from the optimal point.
    
    Args:
    -------------------------
        - x: `np.ndarray`: The input array.
        - xopt: `np.ndarray`: The location of "an invariant" optimum.
        - n_repetitions: `int`: An integer denoting some repetitions (or subgroups)
        - norm_type: `Union[str,int]`: The norm type to compute some distance to penalize even more not being close to the optimum.
    
    Returns:
    -------------------------
        - A `float`with the computation of the penalty
    """
    
    # Get the dimension (hoping the xopt is just a one-dimensional one)
    dim:int = xopt.size
    
    # Set a list to store the computed norm
    norm_list:List[float]=[]
    
    # Get some general indices 
    gen_indices:np.ndarray = np.arange(start=0,stop=dim,dtype=int)
    
    # Regroup the indices into groups
    group_indices:np.ndarray = gen_indices.reshape(
                                                    (n_repetitions,-1)
                                                  )
    for k, shuff in enumerate(permutations(range(n_repetitions),
                                           n_repetitions)):
        
        
        shuff_indices = group_indices[np.asarray(shuff),:].ravel()
        
        shuff_x_opt = xopt[shuff_indices]
        
        # Compute the norm and append to the list
        norm_list.append(np.linalg.norm(x-shuff_x_opt,ord=norm_type))
        
    
    
    return min(norm_list)
    

class GeneralizedGallagherFunctionWithRepetitions(RealSingleObjective):

    r"""
        This is an extension of the Gallagher function by taking the developments from the BBOB-COCO Testbench
        (see: https://coco-platform.org/testsuites/bbob/functions/f21.html) as well as the original paper
        from Gallagher & Yuan ("A General-Purpose Tunable Landscape Generator", 
        IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 10, NO. 5, OCTOBER 2006).

        To follow the other benchmarks, this class extends the IOH framework:
        see: https://iohprofiler.github.io/IOHexperimenter/
    """
    
    def __init__(self,
                 dim:int, 
                 n_repetitions:int,
                 num_peaks:Optional[int]=21, 
                 instance:Optional[int] =1,
                 norm_type:Optional[Union[str,int]]=2)->None:
        
        r"""
        The main initializer of the class.

        Args:
        -------------
      
        - x: Input vector of shape (dim,).
        - dim: `int`: Dimensionality of the input space.
        - n_repetitions: `int`: The number of repetitions (possible permutations) denoting the permutation invariance of the function.
        - num_peaks: `int`: Number of Gaussian peaks.
        - instance: `Optional[int]`; an instance of np.random.RandomState for reproducibility.
        - norm_type: `Optional[Union[str,int]]`: The norm to compute a penalty for not being close to the optimum value
    """
        
        # Input checks
        if not (num_peaks >=1 and isinstance(num_peaks,int)):
            raise ValueError("The number of peaks should be greater than 1")
        
        if not (dim >=1 and isinstance(dim,int)):
            raise ValueError("The dimension of the problem should be a positive integer")
        
        if not (n_repetitions >=1 and isinstance(n_repetitions,int)):
            raise ValueError("The number of repetitions should be a positive integer")
        
        # Check the divisibility of the dimension and repetitions
        if not np.remainder(dim,n_repetitions)==0:
            raise ValueError("The number of repetitions should be a divisor of the problem dimensionality!")
        
        #Instantiate a random state given the instance
        random_state = np.random.RandomState(instance)

        # Store the norm type
        self._norm_type:Union[int,str] = norm_type

        # Store the number of repetitions
        self.__n_repetitions:int = n_repetitions

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

            # Ensure the positive definiteness
            cov = np.dot(cov.T,cov)

            # "Normalize the covariance"
            cov = cov/n_repetitions**2

            # Perform the computation 
            q, r = np.linalg.eigh(cov)
            self._q_list.append(q)
            self._r_list.append(r)
        
        name = "Generalized Gallagher Function with {0} peaks and {1} repetitions".format(num_peaks,n_repetitions)
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
            cur_peak_height = self._peak_heights[i]
            cur_peak_location:np.ndarray = self._peak_locations[i]
            cur_eigenvec:np.matrix = self._r_list[i]
            cur_eigenvals:np.ndarray = self._q_list[i]

            if self.n_repetitions == 1:
                diff = (x - cur_peak_location).reshape(-1,1)
                exponent = (-(1/(2*dim)) * (diff.T @ ( cur_eigenvec @ np.diag(np.divide(1,cur_eigenvals) ) @  cur_eigenvec.T )   @ diff)).ravel()[0]
                value = cur_peak_height * np.exp(exponent)
                values.append(value)
            else:
                
                # Get some general indices 
                gen_indices:np.ndarray = np.arange(start=0,stop=dim,dtype=int)
                
                # Regroup the indices into groups
                group_indices:np.ndarray = gen_indices.reshape(
                                                        (self.n_repetitions,-1)
                                                    
                                                    )
                for k, shuff in enumerate(permutations(range(self.n_repetitions),
                                                       self.n_repetitions)):
                    
                    
                    shuff_indices = group_indices[np.asarray(shuff),:].ravel()
                    
                    
                    shuff_peak_location = cur_peak_location[shuff_indices]
                    shuff_eigenvals = cur_eigenvals[shuff_indices]
                    
                    shuff_eigenvec = cur_eigenvec[:,shuff_indices]
                    shuff_eigenvec = shuff_eigenvec[shuff_indices,:]
                    
                    diff = (x -shuff_peak_location).reshape(-1,1)
                    exponent = (-(1/(2*dim)) * (diff.T @ ( shuff_eigenvec @ np.diag(np.divide(1,shuff_eigenvals) ) @  shuff_eigenvec.T )   @ diff)).ravel()[0]
                    value =  cur_peak_height* np.exp(exponent)
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

        return self._compute_value(x) + _fopt(x=x,xopt=self._peak_locations[0],norm_type=self._norm_type,n_repetitions=self.n_repetitions) + _fpen(x)
    
    def create(self, id, iid, dim):
        raise NotImplementedError
    
    @property
    def n_repetitions(self)->int:
        r"""
        Returns the number of repetitions loaded
        """
        return self.__n_repetitions
    
    @n_repetitions.setter
    def n_repetitions(self,new_input:int)->None:
        r"""
        This is the setter of the number of repetitions
        """
        if not (isinstance(new_input,int) and new_input >=1):
            raise ValueError("The number of repetitions is not a positive integer")
        
        if not np.remainder(self._peak_locations.shape[1],new_input)==0:
            raise ValueError("The number of repetitions should be a divisor of the problem dimensionality!")
        
        # Assign the new number of repetitions
        self.__n_repetitions = new_input
    
    
    def group_size(self)->int:
        r"""
        Returns the size of each group generated by the number of repetitions
        """
        
        return int(self._peak_locations.shape[1]/self.n_repetitions)

    def __del__(self, *args, **kwargs):

        self.detach_logger()



        
        
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
