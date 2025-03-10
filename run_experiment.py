from functools import partial
import random
from wrapper import wrapopt, Abstract_Optimizer_Wrapper
from my_logger import MyIOHFormatOnEveryEvaluationLogger2
import sys
import os
import json


from typing import Callable

import ioh

import numpy as np
from numpy import ndarray
import copy
import time
from typing import List, Tuple, Optional, Union
#from warnings import deprecated
from datetime import timedelta


def decide_doe_size(dim):
    return dim


def decide_total_budget(dim, doe_size):
    return min(100 * dim , 1000)

### ============================================
### CONSTANTS
### ============================================
TRIGGER = ioh.logger.trigger.Each(10)
ONIMPROVEMENT = ioh.logger.trigger.ON_IMPROVEMENT
ALWAYS = ioh.logger.trigger.ALWAYS

### DEPRECATED CLASS ###
class AlgorithmWrapper:
    def __init__(self, 
                 seed:int):
        self.opt = None
        self.seed = seed


    def __call__(self, 
                 optimizer_name:str, 
                 f:ioh.problem.RealSingleObjective, 
                 fid, iid, dim, sample_zero):
        self.dim = dim
        self.optimizer_name = optimizer_name
        doe_size = decide_doe_size(self.dim)
        total_budget = decide_total_budget(self.dim, doe_size)
        self.opt = wrapopt(
            optimizer_name, f, self.dim, total_budget, doe_size, self.seed, sample_zero)
        self.opt.run()


    # @property
    # def lower_space_dim(self) -> int:
    #     if self.optimizer_name == 'BO':
    #         return self.dim
    #     return self.opt.get_lower_space_dimensionality()
    #
    # @property
    # def extracted_information(self) -> float:
    #     if self.optimizer_name == 'BO':
    #         return 1.0
    #     return self.opt.get_extracted_information()
    #
    # @property
    # def kernel_config(self) -> str:
    #     return self.opt._pca.get_kernel_parameters()
    #
    # @property
    # def out_of_the_box_solutions(self) -> int:
    #     return self.opt.out_solutions

    @property
    def acq_opt_time(self) -> float:
        return self.opt.get_acq_time()

    @property
    def model_fit_time(self) -> float:
        return self.opt.get_mode_time()

    @property
    def cum_iteration_time(self) -> float:
        return self.opt.get_iter_time()
    
    

def logger_func(folder_name,my_optimizer_name,sample_zero:bool,**kwargs)->ioh.logger.Analyzer:
    # Set up the logger
    triggers = [ONIMPROVEMENT,
                     TRIGGER]
    
    alg_info:str = ""

    if sample_zero:
        alg_info = "with sampling zero vector"
    
    l = ioh.logger.Analyzer(triggers=triggers,
                    additional_properties=[ioh.logger.property.RAWYBEST],
                    root=os.getcwd(),
                    folder_name=folder_name,
                    algorithm_name=my_optimizer_name,
                    algorithm_info=alg_info,
                    store_positions=False)
    
    return l

def logger_func_2(folder_name,my_optimizer_name,sample_zero:bool,**kwargs)->ioh.logger.Analyzer:
    # Set up the logger
    triggers = [ALWAYS]
    
    alg_info:str = ""

    if sample_zero:
        alg_info = "with sampling zero vector"
    
    l = MyIOHFormatOnEveryEvaluationLogger2(triggers=triggers,
                    properties=[],
                    root=os.getcwd(),
                    folder_name=folder_name,
                    algorithm_name=my_optimizer_name,
                    algorithm_info=alg_info)
    
    return l

def run_particular_experiment(**kwargs):
    
    
    my_optimizer_name = str(kwargs.pop('opt',"BO_botorch")).strip()
    fid = int(kwargs.pop('fid',1)) # Run the sphere function as default
    iid = int(kwargs.pop('iid',0)) # Run the instance 0 as default
    dim = int(kwargs.pop('dim',5)) # Run dimension 5 as a default
    seed = int(kwargs.pop('seed',43)) # Use seed 43 as a default
    rep = int(kwargs.pop('rep',1)) # Set a default of 1 repetition of the experiment
    folder_name = str(kwargs.pop('folder')).strip()
    logger_type = str(kwargs.pop('logger','simple')).lower().strip()
    sample_zero = bool(kwargs.pop('sample_zero',False))

    # Perform some checks
    if len(folder_name) == 0 or folder_name=="":
        raise ValueError("The folder name is empty!")


    p:ioh.problem.RealSingleObjective = ioh.get_problem(fid=fid,
                                                        instance=iid,
                                                        dimension=dim,
                                                        problem_class=ioh.ProblemClass.BBOB)
    
    algorithm:Abstract_Optimizer_Wrapper = wrapopt(optimizer_name=my_optimizer_name,
                                                   func=p,
                                                   ml_dim=dim,
                                                   ml_total_budget=decide_total_budget(dim,decide_doe_size(dim)),
                                                   ml_DoE_size=decide_doe_size(dim),
                                                   random_seed=seed,
                                                   sample_zero=sample_zero
                                                   )
    print("dim = ", dim)

    if logger_type == "complete":
        l = logger_func_2(
            folder_name=folder_name, my_optimizer_name=my_optimizer_name,
            sample_zero=sample_zero)
        #print(f'    Logging to the folder {l.folder_name}')
        #sys.stdout.flush()
        l.watch(algorithm, ['acq_opt_time', 'model_fit_time', 'cum_iteration_time'])
        # try:
        #     l.watch(algorithm,'loss')
        # except Exception as e:
        #     print(e.args)
        #l.watch(p,["loss","best_loss"])
        p.attach_logger(l)

    elif logger_type == "simple":
        # Use the default logger from IOH
        

        # Set up the logger
        l = logger_func(folder_name=folder_name,
                        my_optimizer_name=my_optimizer_name,
                        sample_zero=sample_zero)
        
        p.attach_logger(l)
    else:
        raise AttributeError("The logger type is not either set to 'simple' or 'complete'!",
                             name="logger_type",
                             obj=logger_type)

    
    try:
        algorithm.run()
    
    except KeyboardInterrupt as e:
        print("---Keyboard Cancellation---",
              e.args)
        if isinstance(l,MyIOHFormatOnEveryEvaluationLogger2):
            l.finish_logging()
    except ValueError as e:
        print("---Value error---",
              e.args)
        if isinstance(l,MyIOHFormatOnEveryEvaluationLogger2):
            l.finish_logging()

    except Exception as e:
        print("---Unexpected error---",
              e.args)
        if isinstance(l,MyIOHFormatOnEveryEvaluationLogger2):
            l.finish_logging()
    else:
        p.detach_logger()
                    

def run_experiment():
    if len(sys.argv) == 1:
        print('No configs given')
        return
    with open(sys.argv[1]) as f:
        m = json.load(f)
    print(f'Running with config {m} ...')
    start = time.process_time()
    run_particular_experiment(**m)
    end = time.process_time()
    sec = int(round(end - start))
    x = str(timedelta(seconds=sec)).split(':')
    print(
        f'    Done in {sec} seconds. Which is {x[0]} hours, {x[1]} minutes and {x[2]} seconds')


if __name__ == '__main__':
    run_experiment()
