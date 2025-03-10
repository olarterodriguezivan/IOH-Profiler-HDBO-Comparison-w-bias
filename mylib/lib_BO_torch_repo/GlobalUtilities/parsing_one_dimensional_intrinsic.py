import os
import sys
from argparse import ArgumentParser, Namespace, ArgumentTypeError, ArgumentError
from typing import List, NamedTuple, Tuple, Union
from Algorithms import Pure_CMA_ES, RejectionSamplingCMA_ES, SpecialInjectionCMA_ES
from IOH_Wrappers import ModularNonBBOBProblem, ModularNonBBOBProblemWithSubspace

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_positive2(value):
    r"""
    This is to check the value is positive and not zero
    """
    # Convert the value to a float
    try:
        value = float(value)
    except ValueError:
        raise ArgumentTypeError("%s is an invalid positive float value" % value)
    
    if value <= 0.00:
        raise ArgumentTypeError("%s is an invalid positive float value" % value)
    return value

def check_boolean(value:Union[bool,int,str]): 
    if isinstance(value,int):
        ivalue = bool(value)
    elif isinstance(value,bool):
        ivalue = value
    elif isinstance(value,str):
        if value.strip() == "0":
            ivalue = False
        elif value.strip() == "1":
            ivalue = True
        elif value.strip().lower() == "true":
            ivalue = True
        elif value.strip().lower() == "false":
            ivalue = False
        else:
            raise ArgumentError()
    else:
        raise ArgumentTypeError("%s is an invalid value" % value)
    
    return ivalue
    



def parse_args(args)-> Namespace:
    r"""
    Define a CLI parser and parse command line arguments

    Args:
    ---------
    args: command line arguments

    Returns:
    ---------
    Namespace: parsed command line arguments

    """
     
    parser = ArgumentParser(
        description=r"""Run modified BBOB problems with CMA-ES algorithms 
                        to compare the performance of these when optimizing 
                        the problems with permutation invariant landscapes""",
                    add_help=True,
                    )
    parser.add_argument("--run", 
                        type=check_positive, 
                        default=1,
                        help="Run Number")
    parser.add_argument("--seed", 
                        type=check_positive, 
                        default=43, 
                        help="Random seed")
    parser.add_argument("--budget", 
                        type=check_positive, 
                        default=5000, 
                        help="Budget")
    parser.add_argument("--algorithm", 
                        type=str, 
                        default="pure",
                        choices=["pure", "rejection_sampling", "injection"], 
                        help="CMA-ES Algorithm to run")
    parser.add_argument("--fid",
                        type=int,
                        default=11,
                        choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
                        help="Function ID")
    parser.add_argument("--repetitions",
                        type=int,
                        default=2,
                        help="Number of repetitions")
    parser.add_argument("--invariance_mode",
                        type=str,
                        default="average",
                        choices=["average","maximum"],
                        help="Invariance mode")
    parser.add_argument("--suite",
                        type=str,
                        default="NonBBOB",
                        choices=["NonBBOB","NonBBOBWithSubspace"],
                        help="Suite")
    parser.add_argument("--step_size",
                        type=float,
                        default=2.5,
                        help="Step size for CMA-ES initial parameter")
    parser.add_argument("--restarts",
                        type=check_positive,
                        default=1,
                        help="Number of restarts for CMA-ES")
    parser.add_argument("--root_folder",
                        type=str,
                        default=os.getcwd(),
                        help="Root folder to store data")
    parser.add_argument("--cma_active",
                        type=check_boolean,
                        default=True,
                        help="Use Active CMA-ES or not")
    parser.add_argument("--max_loop",
                        type=int,
                        default=1000,
                        help="Maximum number of loops to seek for rejection sampling")
    parser.add_argument("--verbose",
                        "-v",
                        type=check_boolean,
                        default=False,
                        help="Verbose mode")
    
    parser.add_argument("--shrinkage_factor",
                        "-sf",
                        type=float,
                        default=0.8,
                        help="Shrinkage factor for definition of Rejection close to Tabu Regions")
    parser.add_argument("--rejection_factor_c",
                        '-rfc',
                        type=check_positive2,
                        default=2,
                        help="A parameter to define the rejection region. ",
                        )
    parser.add_argument("--tabu_active",
                        "-ta",
                        type=check_boolean,
                        default=False,
                        help="Use Tabu Search or not")

    parser.add_argument("--n_evals_hill_valley",
                        '-nehv',
                        type=check_positive,
                        default=5,
                        help="Number of evaluations to make for the hill-valley test. ",
                        )
    
    parser.add_argument("--consider_intrinsic_dimension",
                        '-cid',
                        type=check_boolean,
                        default=True,
                        help="A handler to consider intrinsic Dimension when processing Tabu Regions. ",
                        )
    
    pars = parser.parse_args(args)

    
    return pars

_suite_mapper = {
    "NonBBOB": ModularNonBBOBProblem,
    "NonBBOBWithSubspace": ModularNonBBOBProblemWithSubspace
}

_alg_mapper = {
    "pure": Pure_CMA_ES,
    "rejection-sampling": RejectionSamplingCMA_ES,
    "injection": SpecialInjectionCMA_ES
}


    
