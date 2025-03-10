from Generalized_Gallagher_Function import GeneralizedGallagherFunction
from Generalized_Gallagher_Function_with_repetitions import GeneralizedGallagherFunctionWithRepetitions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import ioh
import os
from cma import fmin, CMAEvolutionStrategy, CMAOptions


instance = 12
run_e=80

ioh_prob = GeneralizedGallagherFunctionWithRepetitions(num_peaks=50,
                                        dim=2,
                                        n_repetitions=2,
                                        instance=instance,
                                        norm_type=1)

print(ioh_prob.optimum.x,ioh_prob.optimum.y)
x_ = np.linspace(-5,5,num=100,endpoint=True)
y_ = np.linspace(-5,5,num=100,endpoint=True)

xx, yy = np.meshgrid(x_,y_)

zz = np.zeros_like(xx)

for ii in range(np.shape(xx)[0]):
    for jj in range(np.shape(xx)[1]):
        zz[ii,jj] = ioh_prob([xx[ii,jj],yy[ii,jj]])


fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(xx, yy, zz, 20, cmap=plt.cm.bone)
fig1.colorbar(CS, ax=ax2, shrink=0.9)

plt.show()

ioh_prob.reset()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(xx, yy, zz, cmap=plt.cm.bone,
                    linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel(f"$x_{0}$")
ax.set_ylabel(f"$x_{1}$")
ax.set_zlabel(f"$f(x_{0},x_{1})$")


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

ioh_prob.reset()

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

# class RandomSearch:
#     'Simple random search algorithm'
#     def __init__(self, n: int, length: float = 0.0):
#         self.n: int = n
#         self.length: float = length
        
#     def __call__(self, problem: ioh.problem.RealSingleObjective) -> None:
#         'Evaluate the problem n times with a randomly generated solution'
        
#         for _ in range(self.n):
#             # We can use the problems bounds accessor to get information about the problem bounds
#             x = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
#             self.length = np.linalg.norm(x)
            
#             problem(x)

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
fmin(ioh_prob,x0=minimizer,sigma0=None,restarts=10,options=inopts,bipop=True)
#obj:ioh.iohcpp.RealConstraint = ioh_prob.constraints[0]
#print(type(obj),obj.is_feasible(x_init),obj(x_init))

#cma.fmin2()
#es = cma.evolution_strategy.fmin2(objective_function=ioh_prob,x0=x_init.tolist(),sigma0=1.25,eval_initial_x=True,restarts=10,options=opts)
#es = cma.evolution_strategy.fmin_con2(objective_function=ioh_prob,x0=x_init.tolist(),sigma0=0.25,constraints=,eval_initial_x=True,restarts=10,options=opts)


# print(ioh_prob.constraints[0].is_feasible(x_init),
#       ioh_prob.constraints[1].is_feasible(x_init),
#       ioh_prob.constraints[2].is_feasible(x_init),
#       ioh_prob.constraints[3].is_feasible(x_init))




# If we want to perform multiple runs with the same objective function, after every run, the problem has to be reset, 
# such that the internal state reflects the current run.
# def run_experiment(problem:ioh.problem.RealSingleObjective, algorithm:cma.CMAEvolutionStrategy, n_runs=5):
#     for run in range(n_runs):
        
#         # Run the algorithm on the problem
#         algorithm.optimize(objective_fct=problem,iterations=12,verb_disp=None)

#         # print the best found for this run
#         print(f"run: {run+1} - best found:{problem.state.current_best.y: .3f}")

#         # Reset the problem
#         problem.reset()


#run_experiment(problem=ioh_prob,algorithm=es,n_runs=1)

logger.close()
ioh_prob.detach_logger()