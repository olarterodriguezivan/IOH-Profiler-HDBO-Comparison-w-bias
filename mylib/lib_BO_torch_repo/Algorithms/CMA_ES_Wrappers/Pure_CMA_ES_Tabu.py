r"""
    This module stores the different wrappers for CMA-ES.
    This is an extension of Nikolaus Hansen's et. al. library to perform CMA-ES
    and work within this framework.

    It appends the work from De Nobel et. al. (2024) in regards of Tabu Areas for 
    restarts. See "Avoiding Redundant Restarts in  Multimodal Global Optimization".
"""

### --------------------------------------
### MODULE PROPERTIES
### --------------------------------------

__author__ = ["Iván Olarte Rodríguez"]


import numpy as np # Import Numpy
from ..AbstractAlgorithm import AbstractAlgorithm
from cma import CMAOptions, CMAEvolutionStrategy
from cma.utilities import utils
from cma.options_parameters import safe_str
from cma import optimization_tools as ot
from .Pure_CMA_ES import Pure_CMA_ES
from typing import Union, Callable, List, Optional, Tuple
from ioh.iohcpp.problem import RealSingleObjective
from ..utils.utilities import hill_valley_test, hill_valley_test_2
from itertools import permutations
from scipy.special import gamma as gamma_func
import warnings
import time


class Pure_CMA_ES_Tabu(Pure_CMA_ES):

    def __init__(self,
                 x0:Union[np.ndarray,Callable],
                 sigma0:float,
                 budget:int,
                 inopts:Optional[Union[CMAOptions,dict]]=None,
                 random_seed:Optional[int]=42,
                 **kwargs):
        r""" This is the extended constructor for the pure CMA_ES with Tabu Regions.

        Args:
        -----------
        - x0: An initial point to start with CMA-ES
        - sigma0: The initial step size
        - budget: Number of maximum number of evaluations performed to the problem
        - inopts: The initial options required for CMA-ES
        - **kwargs: keyword arguments to pass to superclasses
        """

        # Use the superclass from Pure_CMA_ES class
        super().__init__(x0,sigma0,budget,inopts,random_seed,**kwargs)

        # Initialize the tabu triplets
        self.__tabu_triplets = []


    def __str__(self):
        return "Pure CMA-ES Optimizer with Tabu zones implementation"
    

    def __call__(self, 
                 problem:Union[RealSingleObjective,Callable],
                 dim:Optional[int]=-1, 
                 bounds:Optional[np.ndarray]=None,
                 shrinkage_factor:Optional[float]=0.5,
                 intrinsic_dimension:Optional[int] = None,
                 rejection_factor_c:Optional[float] = 200.0,
                 n_evals_hill_valley:Optional[int] = 10,
                 **kwargs)->None:
        
        r"""
        This is the modified `__call__` function to implement the
        CMA-ES Algorithm with Tabu Points restarts. 

        Args:
        ------------
        - problem: `Callable`: A callable object (preferably inheriting from IOH), to evaluate a single objective function.
        - dim: `Optional[int]`: An indicator of the dimension of the problem.
        - bounds: `Optional[np.ndarray]`: An array with the bounds of the problem.
        - intrinsic_dimension: `Optional[int]`: An integer denoting the intrinsic dimension of the problem
                                            i.e. The symmetries of the problem.
        - shrinkage_factor: `Optional[float]`: The shrinkage factor to be applied to the rejection radius.
        - rejection_factor_c: `Optional[float]`: The rejection factor to be applied to the rejection radius.
        - n_evals_hill_valley: `Optional[int]`: The number of evaluations to perform in the Hill Valley Test.
        """
        
        # Call the Abstract superclass to check on the problem type and 
        # set the dimension and bounds
        super(Pure_CMA_ES,self).__call__(problem, dim, bounds, **kwargs)
        
        # Ensure the shrinkage factor is between bounds
        if not (shrinkage_factor > 0 and shrinkage_factor < 1):
            raise ValueError("The shrinkage factor must be between 0 and 1")
        
        # Check the intrinsic dimension
        if not ((isinstance(intrinsic_dimension,int) and intrinsic_dimension > 0) or intrinsic_dimension is None):
            raise ValueError("The intrinsic dimension must be a positive integer or None")
        
        # Check the rejection factor
        if not (rejection_factor_c > 0):
            raise ValueError("The rejection factor must be a positive float")
        
        # Check the number of evaluations in the Hill Valley Test

        if not (n_evals_hill_valley > 0 or isinstance(n_evals_hill_valley,int)):
            raise ValueError("The number of evaluations in the Hill Valley Test must be a positive integer")

        

        # Get definitions from kwargs
        restarts:int = kwargs.pop("restarts",0)
        restart_from_best:str = kwargs.pop("restart_from_best","False")
        incpopsize:int = kwargs.pop("incpopsize",2)
        bipop:bool = kwargs.pop("bipop",False)
        callback:Callable = kwargs.pop("callback",None)
        noise_kappa_exponent = kwargs.pop("noise_kappa_exponent",None)
        eval_initial_x:bool = kwargs.pop("eval_initial_x",True)

        

        # Adjust the bounds setting to be fed into the definition of Hansen's CMA-ES
        h_bounds = [self.bounds[:,0].ravel().tolist(), self.bounds[:,1].ravel().tolist()]

        # set the bounds into the options
        self.inopts.set("bounds",h_bounds)


        # Adjust the verbosity
        if self.verbose:
            self.inopts.set("verbose",1)

        # Check the dimensionality and x0 compatibility
        if not self.x0.size == self.dimension:
            # Adjust the dimension and warn the user
            warnings.warn(f"The dimension of the initial point is different from the problem dimension." \
                           "Adjusting the dimension to {self.dimension}")
            if self.dimension < self.x0.size:
                self.x0 = np.array(self.x0[:self.dimension])
            else:
                self.x0 = np.hstack([self.x0,np.zeros(self.dimension-self.x0.size)])

        # Set the options from the saved configuration
        opts = self.inopts
        ###--------------------------------------
        ### This is a direct copy from fmin method
        ###--------------------------------------

        if 1 < 3:  # try: # pass on KeyboardInterrupt

            fmin_options = locals().copy()  # archive original options
            del fmin_options['problem']
            del fmin_options['dim']
            del fmin_options['kwargs']
            del fmin_options['bounds']
            del fmin_options['self']
            del fmin_options['shrinkage_factor']
            del fmin_options['intrinsic_dimension']
            del fmin_options['rejection_factor_c']
            

            if callback is None:
                callback = []
            elif callable(callback):
                callback = [callback]

            # BIPOP-related variables:
            runs_with_small = 0
            small_i = []
            large_i = []
            popsize0 = None  # to be evaluated after the first iteration
            maxiter0 = None  # to be evaluated after the first iteration
            base_evals = 0

            irun = 0
            best = ot.BestSolution()
            all_stoppings = []
            while True:  # restart loop
                sigma_factor = 1

                # Adjust the population according to BIPOP after a restart.
                if not bipop:
                    # BIPOP not in use, simply double the previous population
                    # on restart.
                    if irun > 0:
                        popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                        opts['popsize'] = popsize0 * popsize_multiplier

                elif irun == 0:
                    # Initial run is with "normal" population size; it is
                    # the large population before first doubling, but its
                    # budget accounting is the same as in case of small
                    # population.
                    poptype = 'small'

                elif sum(small_i) < sum(large_i):
                    # An interweaved run with small population size
                    poptype = 'small'
                    if 11 < 3:  # not needed when compared to irun - runs_with_small
                        restarts += 1  # A small restart doesn't count in the total
                    runs_with_small += 1  # _Before_ it's used in popsize_lastlarge

                    sigma_factor = 0.01**np.random.uniform()  # Local search
                    popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                    opts['popsize'] = np.floor(popsize0 * popsize_multiplier**(np.random.uniform()**2))
                    opts['maxiter'] = min(maxiter0, 0.5 * sum(large_i) / opts['popsize'])
                    # print('small basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

                else:
                    # A run with large population size; the population
                    # doubling is implicit with incpopsize.
                    poptype = 'large'

                    popsize_multiplier = fmin_options['incpopsize']**(irun - runs_with_small)
                    opts['popsize'] = popsize0 * popsize_multiplier
                    opts['maxiter'] = maxiter0
                    # print('large basemul %s --> %s; maxiter %s' % (popsize_multiplier, opts['popsize'], opts['maxiter']))

                
                if irun and eval(str(fmin_options['restart_from_best'])):
                    utils.print_warning('CAVE: restart_from_best is often not useful',
                                        verbose=opts['verbose'])
                    es = CMAEvolutionStrategy(best.x, sigma_factor * self.sigma0, opts)
                else:
                    # Perform the sorting function before inserting the initial point
                    

                    if irun==0:
                        x0 = self.x0.ravel()
                        es = CMAEvolutionStrategy(x0, sigma_factor * self.sigma0, opts)
                    else:
                        rng = np.random.default_rng(opts['seed']) 
                        x0 = rng.uniform(0,1,self.dimension).ravel()
                        
                        for idx_,elem in enumerate(x0):
                            x0[idx_] = self.bounds[idx_,0] + elem*(self.bounds[idx_,1]-self.bounds[idx_,0])
                        
                        del idx_,elem
                        # Start up the CMA-ES Evolution Strategy
                        es = CMAEvolutionStrategy(x0, sigma_factor * self.sigma0, opts)
                        
                # return opts, es
                if callable(problem) and (
                    eval_initial_x
                    or es.opts['CMA_elitist'] == 'initial'
                    or (es.opts['CMA_elitist'] and
                                eval_initial_x is None)):
                    

                    x = es.gp.pheno(es.mean,
                                    into_bounds=es.boundary_handler.repair,
                                    archive=es.sent_solutions)
                    es.f0 = problem(x)
                    es.best.update([x], es.sent_solutions,
                                [es.f0], 1)
                    es.countevals += 1
                es.objective_function = problem  # only for the record

                opts = es.opts  # processed options, unambiguous
                # a hack:
                fmin_opts = CMAOptions("unchecked", **fmin_options.copy())
                for k in fmin_opts:
                    # locals() cannot be modified directly, exec won't work
                    # in 3.x, therefore
                    fmin_opts.eval(k, loc={'N': es.N,
                                        'popsize': opts['popsize']},
                                correct_key=False)

                es.logger.append = opts['verb_append'] or es.countiter > 0 or irun > 0
                # es.logger is "the same" logger, because the "identity"
                # is only determined by the `verb_filenameprefix` option
                logger = es.logger  # shortcut
                try:
                    logger.persistent_communication_dict.update(
                        {'variable_annotations':
                        problem.variable_annotations})
                except AttributeError:
                    pass


                # Set noise handling
                #noisehandler = ot.NoiseHandler(es.N, 0)  # switched off
                #noise_handling = False
                #es.noise_handler = noisehandler

        
                while not es.stop():  # iteration loop
                    # X, fit = eval_in_parallel(lambda: es.ask(1)[0], es.popsize, args, repetitions=noisehandler.evaluations-1)
                    # X, fit = es.ask_and_eval(problem,
                    #                         args=[], gradf=None,
                    #                         evaluations=noisehandler.evaluations,
                    #                         aggregation=np.median,
                    #                         parallel_mode=False)  # treats NaN with resampling if not parallel_mode
                    # TODO: check args and in case use args=(noisehandler.evaluations, )

                    
                    # Start the X array (as an empty list)
                    X:list = []
                    # Iteration counter
                    n_sampling:int = 0

                    # ask values
                    n_ask_vals:int = es.popsize

                    # ASK Step (Perform loops until reaching actual pupulation size)
                    while len(X) < es.popsize:
                        # Get the sample from the CMA-ES
                        x_samples = es.ask(number=n_ask_vals)

                        # Check if each of the sampled points is rejected
                        for x_samp in x_samples:
                            if self._reject_point(es,x_samp,shrinkage_factor,n_sampling,rejection_factor_c,irun,intrinsic_dimension):
                                # Add the rejected point
                                n_sampling += 1
                            else:
                                # Append the sample to the list
                                X.append(x_samp)
                        
                        # Update the number of asked values
                        n_ask_vals = es.popsize - len(X)
          

                    # FIT
                    fit = [problem(x_i) for x_i in X]


                    # if es.opts['verbose'] > 4:  # may be undesirable with dynamic fitness (e.g. Augmented Lagrangian)
                    #     if es.countiter < 2 or min(fit) <= es.best.last.f:
                    #         degrading_iterations_count = 0  # comes first to avoid code check complaint
                    #     else:  # min(fit) > es.best.last.f:
                    #         degrading_iterations_count += 1
                    #         if degrading_iterations_count > 4:
                    #             utils.print_message('%d f-degrading iterations (set verbose<=4 to suppress)'
                    #                                 % degrading_iterations_count,
                    #                                 iteration=es.countiter)


                                             
                    es.tell(X, fit, check_points=None)  # prepare for next iteration

                    for f in callback:
                        f is None or f(es)
                    es.disp()
                    logger.add(# more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                            modulo=1 if es.stop() and logger.modulo else None)
                    # if (opts['verb_log'] and opts['verb_plot'] and
                    #     (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
                    #     logger.plot(324)
                

                

                # end while not es.stop
                if opts['eval_final_mean'] and callable(problem):
                    mean_pheno = es.gp.pheno(es.mean,
                                            into_bounds=es.boundary_handler.repair,
                                            archive=es.sent_solutions)
                    fmean = problem(mean_pheno)
                    es.countevals += 1
                    es.best.update([mean_pheno], es.sent_solutions, [fmean], es.countevals)

                best.update(es.best, es.sent_solutions)  # in restarted case
                # es.best.update(best)

                this_evals = es.countevals - base_evals
                base_evals = es.countevals

                # Modify the tabu points archive
                if self.num_tabu_points() == 0:
                    # Modify the tabu points archive
                    self._check_tabu_points_archive(X[-1],fit[-1],problem,intrinsic_dimension,n_evals_hill_valley,True)
                else:
                    # Add evaluations and modify the Tabu Archive (given that the number of evaluations is not reached)
                    if not es.countevals >= self.budget:
                        es.countevals += self._check_tabu_points_archive_2(X[-1],fit[-1],problem,intrinsic_dimension,n_evals_hill_valley, False)
                    else:
                        es.countevals += self._check_tabu_points_archive_2(X[-1],fit[-1],problem,intrinsic_dimension,n_evals_hill_valley, False)
      

                # BIPOP stats update

                if irun == 0:
                    popsize0 = opts['popsize']
                    maxiter0 = opts['maxiter']
                    # XXX: This might be a bug? Reproduced from Matlab
                    # small_i.append(this_evals)

                if bipop:
                    if poptype == 'small':
                        small_i.append(this_evals)
                    else:  # poptype == 'large'
                        large_i.append(this_evals)

                # final message
                if opts['verb_disp']:
                    es.result_pretty(irun, time.asctime(time.localtime()),
                                    best.f)

                irun += 1
                # if irun > fmin_opts['restarts'] or 'ftarget' in es.stop() \
                # if irun > restarts or 'ftarget' in es.stop() \
                all_stoppings.append(dict(es.stop(check=False)))  # keeping the order
                if irun - runs_with_small > fmin_opts['restarts'] or 'ftarget' in es.stop() \
                        or 'maxfevals' in es.stop(check=False) or 'callback' in es.stop(check=False):
                    break
                opts['verb_append'] = es.countevals
                opts['popsize'] = fmin_opts['incpopsize'] * es.sp.popsize  # TODO: use rather options?
                try:
                    opts['seed'] += 1
                except TypeError:
                    pass

            # while irun

            # es.out['best'] = best  # TODO: this is a rather suboptimal type for inspection in the shell
            if irun:
                es.best.update(best)
                # TODO: there should be a better way to communicate the overall best
            
            # Assign  the number of iterations and number of function evaluations to the class
            self.iterations = es.countiter
            self.number_of_function_evaluations = es.countevals
            return es.result + (es.stop(), es, logger)
            ### 4560
            # TODO refine output, can #args be flexible?
            # is this well usable as it is now?
        else:  # except KeyboardInterrupt:  # Exception as e:
            if eval(safe_str(opts['verb_disp'])) > 0:
                print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
            raise KeyboardInterrupt  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit



        # Call the fmin2 function directly
        # res = fmin(
        #     objective_function=problem,
        #     x0=self.x0,
        #     sigma0=self.sigma0,
        #     options = self.inopts,
        #     eval_initial_x=True,
        #     restart_from_best=restart_from_best,
        #     restarts=restarts,
        #     incpopsize=incpopsize,
        #     bipop=bipop,
        #     callback=callback,
        #     noise_kappa_exponent=noise_kappa_exponent
        # )

        # Set the best value
        #self.number_of_function_evaluations = res[3]
        #self.iterations = res[4]
    
    def reset(self):
        # Call the reset method from superclass
        super().reset()

        # Restart the Tabu Triplets
        self.tabu_triplets = []
    
    @property
    def tabu_triplets(self)->List[List[Union[np.ndarray,float,int]]]:
        return self.__tabu_triplets
    
    def num_tabu_points(self)->int:
        r"""
        Returns the number of tabu points.
        """
        return len(self.__tabu_triplets)
    
    @staticmethod
    def compute_rejection_radius(dim:int, V:float)->float:
        r"""
        This is a static method to compute the rejection radius for the tabu points.

        Args:
        ------------
        - dim: `int`: The dimension of the problem
        - V: `float`: The search space coverage given the tabu points.
        """
        return (V**(1.0/dim)/np.sqrt(np.pi))*(gamma_func(dim/2.0+1.0)**(1/dim))
    
    @staticmethod
    def compute_search_space_coverage(n_rep:int, vol:float, c:float, sigma_0:float, R:int)->float:
        r"""
        This is a static method to compute the search space coverage given the tabu points.

        Args:
        -------------
        - n_rep: `int`: The number of repetitions of the Tabu point in the archive.
        - vol: `float`: The volume of the search space.
        - c: `float`: The 'coverage factor' as mentioned by De Nobel et al.
        - sigma_0: `float`: The initial step size of the CMA-ES algorithm.
        - R: `int`: The number of restarts (so far)
        """

        return (n_rep*vol)/(c*sigma_0*R)
    
    def _reject_point(self,
                     evol_strategy:CMAEvolutionStrategy,
                     x_sample:np.ndarray,
                     shrinkage_factor:float,
                     n_repetitions:int,
                     c:float,
                     R:int,
                     intrinsic_dimension:Union[None,int])->bool:
        
        r"""
        This function rejects a point if it is within the "tabu zone".

        Args:
        -------------
        - evol_strategy: `CMAEvolutionStrategy`: The evolution strategy object (from Nikolaus Hansen's CMA-ES library)
        - x_sample: `np.ndarray`: The sample to be evaluated.
        - shrinkage_factor: `float`: The shrinkage factor to be applied.
        - n_repetitions: `int`: The number of repetitions or resamples so far in the loop.
        - c: `float`: The coverage factor as mentioned by de Nobel et al.
        - R: `int`: The number of restarts so far.
        - intrinsic_dimension: `Union[None,int]`: The intrinsic dimension of the problem.

        Returns:
        ------------
        `bool`: A boolean indicating if the point is rejected or not.
                NOTE: True means the point is rejected.
        """

        # Initialize a response 
        resp = False
        # Compute the volume of the search space
        vol_search = self.compute_space_volume()

        if self.num_tabu_points() == 0:
            return False
        
        # Loop all over the points of the archive
        for triplet in self.tabu_triplets:
            
            # Check if the intrinsic dimension is None (Case 1)
            if intrinsic_dimension is None:
                # Compute the Mahalanobis distance between the sample and the triplet
                diff = x_sample - triplet[0]
                dist = evol_strategy.mahalanobis_norm(evol_strategy.gp.geno(diff))
                # Compute the the search space coverage
                V_t = self.compute_search_space_coverage(triplet[2], vol_search, c,self.sigma0, R)
                # Compute the rejection radius
                delta_t = self.compute_rejection_radius(self.dimension, V_t)
                # Check if the distance is less than the scaled rejection radius
                if dist < shrinkage_factor**n_repetitions*delta_t:
                    resp = True
                    break
            # Check if the intrinsic dimension is not None (thus iterating the different permutations)
            else:
                # Resize the triplet in accordance to the intrinsic dimension
                triplet_resize:np.ndarray = triplet[0].reshape(-1,intrinsic_dimension,order='F')
                # Compute the number of groups
                n_groups = triplet_resize.shape[0]
                # Iterate over the intrinsic dimension
                perms = permutations(range(n_groups),n_groups)

                was_rejected:bool = False

                # Loop all over the possible permutations
                for idx_,perm in enumerate(perms):
                    reshaped_triplet = triplet_resize[perm,:].ravel()

                    # Compute the Mahalanobis distance between the sample and the triplet
                    diff = x_sample - reshaped_triplet
                    dist = evol_strategy.mahalanobis_norm(evol_strategy.gp.geno(diff))
                    # Compute the the search space coverage
                    V_t = self.compute_search_space_coverage(triplet[2], vol_search, c,self.sigma0, R)
                    # Compute the rejection radius
                    delta_t = self.compute_rejection_radius(self.dimension, V_t)
                    # Check if the distance is less than the scaled rejection radius
                    if dist < shrinkage_factor**n_repetitions*delta_t:
                        was_rejected = True
                        break
                
                if was_rejected:
                    resp = True
                

        return resp
    
    def _check_tabu_points_archive(self,
                                   last_x_sample:np.ndarray,
                                   last_f_sample:float,
                                   problem:Union[RealSingleObjective,Callable],
                                   intrinsic_dimension:Optional[int]=None,
                                   n_evals_hill_valley:Optional[int]=10,
                                   just_append:Optional[bool]=False,
                                   )->None:
        r"""
        This function is used to check the tabu points archive and update it accordingly given the last sample before the new restart.
        
        Args:
        ------------
        - last_x_sample: `np.ndarray`: The last sample evaluated before the restart.
        - last_f_sample: `float`: The last function evaluation before the restart.
        - problem: `Union[RealSingleObjective,Callable]`: The problem to be evaluated with a `__call__` method implemented evaluating the function.
        - intrinsic_dimension: `Optional[int]`: The intrinsic dimensionality of the problem.
        - n_evals_hill_valley: `Optional[int]`: The number of evaluations to perform in the Hill Valley Test.
        - just_append: `Optional[bool]`: A boolean indicating if the point should be appended to the archive without checking.
        """

        # Case when the archive is empty
        if self.num_tabu_points() == 0 or just_append:
            self.tabu_triplets.append([last_x_sample,last_f_sample,1])
        else:
            # Check if the intrinsic dimension is None (Case 1)
            if intrinsic_dimension is None:
                # Loop all over the points of the archive
                broken_loop = False
                for idx,triplet in enumerate(self.tabu_triplets):
                    # Use the Hill Valley Test to check if the point is in the same basin of attraction
                    if hill_valley_test(last_x_sample,last_f_sample,triplet[0],triplet[1],problem,n_evals_hill_valley):
                        self.tabu_triplets[idx][2] += 1
                        broken_loop = True
                        break
   
                    
                if not broken_loop:
                    self.tabu_triplets.append([last_x_sample,last_f_sample,1])

                
            # Check if the intrinsic dimension is not None (thus iterating the different permutations)
            else:
                # Loop all over the points of the archive
                for idx,triplet in enumerate(self.tabu_triplets):
                    broken_loop = False
                    # Resize the triplet in accordance to the intrinsic dimension
                    triplet_resize:np.ndarray = triplet[0].reshape(-1,intrinsic_dimension,order='F')
                    # Compute the number of groups
                    n_groups = triplet_resize.shape[0]
                    # Iterate over the intrinsic dimension
                    perms = [*permutations(range(n_groups),n_groups)]

                    # Initialise a distance array to check the closest invariant point from the actual sample
                    dist_array:list = []

                    # Loop all over the possible permutations
                    for _,perm in enumerate(perms):
                        reshaped_triplet = triplet_resize[perm,:].ravel()

                        # Compute the distance between the reshaped triplet and the last sample (euclidean distance)
                        actual_dist = np.linalg.norm(reshaped_triplet-np.asarray(last_x_sample,dtype=float),ord=2)

                        # Append the to the distance array
                        dist_array.append(actual_dist)
                    
                    # Get the index of the closest distance
                    idx_ = np.argmin(np.asarray(dist_array))

                    compared_perm = perms[idx_]

                    # Use the Hill Valley Test to check if the point is in the same basin of attraction
                    if hill_valley_test(last_x_sample,last_f_sample,triplet_resize[compared_perm,:].ravel(),triplet[1],problem,n_evals_hill_valley):
                        self.tabu_triplets[idx][2] += 1
                        broken_loop = True
                        break

                if not broken_loop:
                    self.tabu_triplets.append([last_x_sample,last_f_sample,1])
    
    def _check_tabu_points_archive_2(self,
                                   last_x_sample:np.ndarray,
                                   last_f_sample:float,
                                   problem:Union[RealSingleObjective,Callable],
                                   intrinsic_dimension:Optional[int]=None,
                                   n_evals_hill_valley:Optional[int]=10,
                                   just_append:Optional[bool]=False,
                                   )->int:
        r"""
        This function is used to check the tabu points archive and update it accordingly given the last sample before the new restart.
        
        Contrary to the original implementation, this function returns the number of function evaluations performed in the Hill Valley Test.

        Args:
        ------------
        - last_x_sample: `np.ndarray`: The last sample evaluated before the restart.
        - last_f_sample: `float`: The last function evaluation before the restart.
        - problem: `Union[RealSingleObjective,Callable]`: The problem to be evaluated with a `__call__` method implemented evaluating the function.
        - intrinsic_dimension: `Optional[int]`: The intrinsic dimensionality of the problem.
        - n_evals_hill_valley: `Optional[int]`: The number of evaluations to perform in the Hill Valley Test.
        - just_append: `Optional[bool]`: A boolean indicating if the point should be appended to the archive without checking.

        Returns:
        ------------
        `int`: The number of function evaluations performed in the Hill Valley Test.
        """

        # The following variable is used for counting a number of function evaluations
        iter_counter:int = 0

        # Case when the archive is empty
        if self.num_tabu_points() == 0 or just_append:
            # Append the first triplet
            self.tabu_triplets.append([last_x_sample,last_f_sample,1])

            # Return 0 automatically since there is no need to compute iterations
            return iter_counter
        else:
            # Check if the intrinsic dimension is None (Case 1)
            if intrinsic_dimension is None:
                # Loop all over the points of the archive
                broken_loop = False
                for idx,triplet in enumerate(self.tabu_triplets):
                    # Use the Hill Valley Test to check if the point is in the same basin of attraction
                    broken_loop, new_iters = hill_valley_test_2(last_x_sample,last_f_sample,triplet[0],triplet[1],problem,n_evals_hill_valley)
                    if broken_loop:
                        self.tabu_triplets[idx][2] += 1
                        iter_counter += new_iters
                        break
   
                    
                if not broken_loop:
                    self.tabu_triplets.append([last_x_sample,last_f_sample,1])

                
                # Return the number of iterations
                return iter_counter

                
            # Check if the intrinsic dimension is not None (thus iterating the different permutations)
            else:
                # Loop all over the points of the archive
                for idx,triplet in enumerate(self.tabu_triplets):
                    broken_loop = False
                    # Resize the triplet in accordance to the intrinsic dimension
                    triplet_resize:np.ndarray = triplet[0].reshape(-1,intrinsic_dimension,order='F')
                    # Compute the number of groups
                    n_groups = triplet_resize.shape[0]
                    # Iterate over the intrinsic dimension
                    perms = [*permutations(range(n_groups),n_groups)]

                    # Initialise a distance array to check the closest invariant point from the actual sample
                    dist_array:list = []

                    # Loop all over the possible permutations
                    for _,perm in enumerate(perms):
                        reshaped_triplet = triplet_resize[perm,:].ravel()

                        # Compute the distance between the reshaped triplet and the last sample (euclidean distance)
                        actual_dist = np.linalg.norm(reshaped_triplet-np.asarray(last_x_sample,dtype=float),ord=2)

                        # Append the to the distance array
                        dist_array.append(actual_dist)
                    
                    # Get the index of the closest distance
                    idx_ = np.argmin(np.asarray(dist_array))

                    compared_perm = perms[idx_]

                    # Use the Hill Valley Test to check if the point is in the same basin of attraction
                    broken_loop, new_iters = hill_valley_test_2(last_x_sample,last_f_sample,triplet_resize[compared_perm,:].ravel()
                                                                    ,triplet[1],problem,n_evals_hill_valley)
                    if broken_loop:
                        self.tabu_triplets[idx][2] += 1
                        iter_counter += new_iters
                        break


                if not broken_loop:
                    self.tabu_triplets.append([last_x_sample,last_f_sample,1])
                
                # Return the number of iterations
                return iter_counter


            




                     




        