from Generalized_Gallagher_Function import GeneralizedGallagherFunction
from Generalized_Gallagher_Function_with_repetitions import GeneralizedGallagherFunctionWithRepetitions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import ioh
import os
from cma import fmin, CMAEvolutionStrategy, CMAOptions


instance = 20
run_e=92

ioh_prob = GeneralizedGallagherFunctionWithRepetitions(num_peaks=33,
                                        dim=3,
                                        n_repetitions=3,
                                        instance=instance,
                                        norm_type=1)

print(ioh_prob.optimum.x,ioh_prob.optimum.y)

triggers = [
    ioh.logger.trigger.Each(1),
    ioh.logger.trigger.OnImprovement()
]

logger = ioh.logger.Analyzer(
    root=os.getcwd(),                  # Store data in the current working directory
    folder_name=f"./ModularExperiments/Run_{run_e}",       # in a folder named: 'my-experiment'
    algorithm_name="CMA-ES",    # meta-data for the algorithm used to generate these results
    store_positions=True,               # store x-variables in the logged files
    triggers= triggers,

    additional_properties=[
        #ioh.logger.property.CURRENTY,   # The constrained y-value, by default only the untransformed & unconstraint y
                                        # value is logged. 
        ioh.logger.property.RAWYBEST, # Store the raw-best
    ]

)

ioh_prob.attach_logger(logger)


x_init = 10*np.ravel(np.random.rand(1,ioh_prob.meta_data.n_variables))-5
#u#u = SpecialInjectionCMA_ES(x_init,
#                 sigma0=1.25,
#                 budget=1000, random_seed=RANDOM_SEED, verbose = True, inopts=opts,
#                 )
#uu(problem=ioh_prob, restarts=10,sorting_function= sorting_function)

inopts:dict = {"maxfevals":20000,
               "verb_filenameprefix":os.path.join(logger.output_directory,"outcmaes/"),
               "bounds":[-5,5],
               "seed":run_e,
               "CMA_active":False}

minimizer = CMAEvolutionStrategy(x0=x_init,sigma0=2.50,inopts=inopts)
fmin(ioh_prob,x0=minimizer,sigma0=None,restarts=10,options=inopts,bipop=False)


logger.close()
ioh_prob.detach_logger()