r"""
    This module stores the different wrappers for CMA-ES.
    This is an extension of Nikolaus Hansen's et. al. library to perform CMA-ES
    and work within this framework
"""

### --------------------------------------
### MODULE PROPERTIES
### --------------------------------------

__author__ = ["Iván Olarte Rodríguez"]


import numpy as np # Import Numpy
from .Pure_CMA_ES import Pure_CMA_ES
import time
from cma.utilities import utils
import cma
from cma import cma_default_options_
from cma import CMAOptions,  CMAEvolutionStrategy
from cma.options_parameters import safe_str
from cma import optimization_tools as ot
from cma import CMADataLogger
from typing import Union, Callable, List, Optional
import warnings


class RejectionSamplingCMA_ES(Pure_CMA_ES):

    def __init__(self,
                 x0:Union[np.ndarray,Callable],
                 sigma0:float,
                 budget:int,
                 inopts:Optional[Union[CMAOptions,dict]]=None,
                 random_seed:Optional[int]=42,
                 **kwargs):
        
        r""" This is the extended constructor for the pure CMA_ES.

        Args:
        -----------
        - x0: An initial point to start with CMA-ES
        - sigma0: The initial step size
        - budget: Number of maximum number of evaluations performed to the problem
        - inopts: The initial options required for CMA-ES
        - **kwargs: keyword arguments to pass to superclasses
        """

        # Use the superclass from Pure CMA-ES
        super().__init__(x0=x0,
                         sigma0=sigma0,
                         budget=budget,
                         inopts=inopts,
                         random_seed=random_seed,
                         **kwargs)
    

    def __str__(self):
        return "Special Injection CMA-ES Optimizer"
    

    def __call__(self, 
                 problem:Callable,
                 dim:Optional[int]=-1, 
                 bounds:Optional[np.ndarray]=None,
                 sorting_function:Optional[Callable]=None,
                 **kwargs)->None:
        
        r"""
        This is an overload of the call function to perform the CMA-ES optimization.

        Args:
        -----------
        - problem: The problem to be optimized
        - dim: The dimension of the problem
        - bounds: The bounds of the problem
        - sorting_function: The sorting function to be used to sort the samples and shift the permutations.
        - **kwargs: Keyword arguments to pass to the CMA-ES optimizer
        """

        # If the sorting function is not given, then, call the superclass
        if sorting_function is None:
            super().__call__(problem, dim, bounds, **kwargs)

        elif callable(sorting_function):
        # This is to handle if the sorting function complies with the callable type
            # Call the `Pure_CMA_ES` superclass to check on the problem type and 
            # set the dimension and bounds
            super(Pure_CMA_ES,self).__call__(problem, dim, bounds, **kwargs)

            # Get definitions from kwargs
            restarts:int = kwargs.pop("restarts",0)
            restart_from_best:str = kwargs.pop("restart_from_best","False")
            incpopsize:int = kwargs.pop("incpopsize",2)
            bipop:bool = kwargs.pop("bipop",False)
            callback:Callable = kwargs.pop("callback",None)
            noise_kappa_exponent = kwargs.pop("noise_kappa_exponent",None)
            eval_initial_x:bool = kwargs.pop("eval_initial_x",True)
            max_loop:int = kwargs.pop("max_loop",100)


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
                del fmin_options['sorting_function']

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
                        if irun==0:
                            x0 = self.x0.ravel()
                            # Sort x0
                            x0,_ = sorting_function([x0])
                            # Start up the CMA-ES Evolution Strategy
                            es = CMAEvolutionStrategy(x0, sigma_factor * self.sigma0, opts)
                        else:
                            rng = np.random.default_rng(opts['seed']) 
                            x0 = rng.uniform(0,1,self.dimension).ravel()
                            
                            for idx_,elem in enumerate(x0):
                                x0[idx_] = self.bounds[idx_,0] + elem*(self.bounds[idx_,1]-self.bounds[idx_,0])
                            
                            # Sort x0
                            x0,_ = sorting_function([x0])
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
                    noisehandler = ot.NoiseHandler(es.N, 0)  # switched off
                    noise_handling = False
                    es.noise_handler = noisehandler

               
                    while not es.stop():  # iteration loop
                        # X, fit = eval_in_parallel(lambda: es.ask(1)[0], es.popsize, args, repetitions=noisehandler.evaluations-1)
                        # X, fit = es.ask_and_eval(problem,
                        #                         args=[], gradf=None,
                        #                         evaluations=noisehandler.evaluations,
                        #                         aggregation=np.median,
                        #                         parallel_mode=False)  # treats NaN with resampling if not parallel_mode
                        # TODO: check args and in case use args=(noisehandler.evaluations, )

                        #X_2 = es.ask()
                        # Start X as an empty list
                        X = []
                        counter = 0 # The counter for the number of loops is set to avoid infinite loops
                        n_sampled = es.popsize
                        while len(X) < es.popsize and counter < max_loop:
                            # Get the samples
                            X_2 = es.ask(n_sampled)
                            # Perform the sorting function
                            _, is_sorted = sorting_function(X_2)

                            # Append the samples
                            for idx, elem in enumerate(is_sorted):
                                if elem:
                                    X.append(X_2[idx])

                            # Update the number of samples
                            n_sampled = es.popsize - len(X)
                            counter +=1

                            # Append the solutions
                            if counter==max_loop:
                                for idx, elem in enumerate(X_2):
                                    if idx < n_sampled:
                                        X.append(elem)
                                
                        
                        del X_2, idx, elem
                        # Evaluate the samples
                        fit = [problem(x) for x in X]

                        if es.opts['verbose'] > 4:  # may be undesirable with dynamic fitness (e.g. Augmented Lagrangian)
                            if es.countiter < 2 or min(fit) <= es.best.last.f:
                                degrading_iterations_count = 0  # comes first to avoid code check complaint
                            else:  # min(fit) > es.best.last.f:
                                degrading_iterations_count += 1
                                if degrading_iterations_count > 4:
                                    utils.print_message('%d f-degrading iterations (set verbose<=4 to suppress)'
                                                        % degrading_iterations_count,
                                                        iteration=es.countiter)

                    
                        
                        # Contrary to the Special Injection, the samples are sorted and part 
                        # of the distribution
                        es.tell(X, fit, check_points=None)  # prepare for next iteration

                        for f in callback:
                            f is None or f(es)
                        es.disp()
                        logger.add(# more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                                modulo=1 if es.stop() and logger.modulo else None)
                        if (opts['verb_log'] and opts['verb_plot'] and
                            (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop())):
                            logger.plot(324)

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



        # # Call the fmin2 function directly\
        # res = cma.fmin(
        #     objective_function=problem,
        #     x0=self.x0,
        #     sigma0=self.sigma0,
        #     options = self.inopts,
        #     eval_initial_x=True,
        #     restart_from_best=restart_from_best,
        #     restarts=restarts,
        #     incpopsize=incpopsize,
        #     bipop=bipop,
        #     callback=callback
        # )

        # # Set the best value
        # self.number_of_function_evaluations = res[3]
        # self.iterations = res[4]

        else:
            raise AttributeError("The sorting function should be a callable object",
                                 name="sorting function",
                                 obj=sorting_function)
    
    @property
    def inopts(self)->CMAOptions:
        r"""
        This is an overload of the getter for the inopts property
        """
        return self.__inopts

    @inopts.setter
    def inopts(self, new_inopts:Union[CMAOptions,dict]):
        r"""
        This is an overload of the setter for the inopts property
        """

        if new_inopts is None or (isinstance(new_inopts,list) and len(new_inopts)==0 ):
            # Start the object with 
            self.__inopts = CMAOptions("unchecked",new_inopts).complement()
        
        elif isinstance(new_inopts,dict):
            self.__inopts = CMAOptions("unchecked",**new_inopts.copy()).complement()
        
        elif isinstance(new_inopts,CMAOptions):
            self.__inopts = new_inopts.complement()
        
        else:
            raise AttributeError("The passed update is not a dictionary, empty list or ``CMAOptions` ")
        
        # Set the maximum number of iterations
        self.__inopts.set("maxfevals",self.budget)

        # Set the random seed
        self.__inopts.set("seed",self.random_seed)

        # Do not set to evaluate the final mean
        self.__inopts.set("eval_final_mean",False)



    
