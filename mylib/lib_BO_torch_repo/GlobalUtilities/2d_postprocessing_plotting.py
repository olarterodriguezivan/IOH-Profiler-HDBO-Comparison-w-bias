r"""
This is a module which will be used to generate convergence plots for 2D Modular problems

---------------------
NOTE: 
This is just a template, which will be further developed in the future for other cases
--------------------
"""

__author__ = ["Iván Olarte Rodríguez"]
__institute__ = ["Leiden Institute of Advanced Computer Science"]


# Import libraries/modules
import os, sys
from IOH_parser import IOH_Parser # Parser defined within this context
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, List, Callable, Iterable, Dict

#Converters
__idx_space_dict:dict = {1:"bimodal",
                             2:"multimodal",
                             3:"plateau",
                             4:"ackley",
                             5:"gramacy_lee",
                             6:"levy",
                             7:"rastrigin",
                             8:"schwefel",
                             9:"sum_different_powers",
                             10:"forrester",
                             11:"sphere",
                             12:"alpine",
                             13:"styblinski_tank",
                             14:"salomon"}


### ------------------
### CONSTANTS 
### ------------------

# IMMUTABLE
PROBLEMS_MODULES:tuple = ("NonBBOB","NonBBOBWithSubspace")
ALGORITHMS_MODULES:tuple = ("injection","pure","rejection_sampling")
OUTCMAES_FILENAMES:tuple = ("axlen.dat",
                            "axlencorr.dat",
                            "axlenprec.dat",
                            "fit.dat",
                            "stddev.dat",
                            "xmean.dat",
                            "xrecentbest.dat")
INVARIANCE_MODES:tuple = ("average","maximum")

RELEVANT_OUTCMAES:list = [OUTCMAES_FILENAMES[3],OUTCMAES_FILENAMES[5],OUTCMAES_FILENAMES[6]]
FID_LIMITS:list = [*range(1,15)]

# TODO: Filled by the user
ROOT_MODULAR_EXPERIMENTS:str = os.path.join(os.getcwd(),"ModularExperiments")
ROOT_TWO_DIMENSIONAL_PROBLEMS_REPO:str = "C:/Users/iolar/Documents/Modular_Problems_IOH/Two_Dimensional"


### -------------------
### VARIOUS FUNCTIONS
### -------------------

def _convert_2_path_object(path_:Union[str,Path])->Path:
    r"""
    This is a helper function to use to convert path_strings to Path

    Args:
    --------------------
    - path_: `Union[str,Path]`: A string path to convert

    Returns:
    - A ``Path` object
    """

    if isinstance(path_,Path):
        return path_
    elif isinstance(path_,str):
        return Path(path_)
    else:
        raise AttributeError("given path not an instance of Path or a string", name="path_",obj=path_)

def _return_identifiers_from_file_path(path_:Union[str,Path])->Dict:
    r"""
    This is a function which just identifies all the problem parameters given a path.

    Args:
    -------------------
    - path_: `Union[str,Path]`: A path object identifying a 'Run_' instance

    Returns:
    -------------------
    A dictionary with the keys;
        - func_type:`str`: The function type, which should have two options: ("NonBBOB","NonBBOBWithSubspace")
        - inv_mode:`str` = The invariance mode: ('average', 'maximum)
        - cur_alg:`str` = The algorithm applied to the problem ('pure','injection','rejection_sampling')
        - func_id:`int` = The id alias of the function (so far an integer between 1 to 14)
        - func_name:`str` = The actual name of the function (check the namespace `__idx_space_dict`)
        - run:`int` = The integer mapping a run number
    """

    # Break down the path to extract information regarding the function
    trial_path:Path = _convert_2_path_object(path_)

    # Transform the path to posix ("/")
    trial_path_ = trial_path.as_posix()

    # Split all the datastructure
    data_struct_full:list = trial_path_.split("/")
    data_struct:list = data_struct_full[-5:]

    # Get the function type, invariance mode, function id and run
    func_type:str = data_struct[0]
    inv_mode:str = data_struct[1]
    cur_alg:str = data_struct[2]
    func_id:int = int(data_struct[3].replace("f",""))
    func_name:str = __idx_space_dict[func_id]
    run_:int = int(data_struct[4].replace("Run_",""))


    return {"func_type":func_type,
            "inv_mode":inv_mode,
            "cur_alg":cur_alg,
            "func_id":func_id,
            "func_name":func_name,
            "run":run_}



def append_global_iteration_lambda_outcmaes(outcmaes_func:Callable)->Callable:
    r"""
    This is a function which receives an outcmaes importing function (ones which takes a saved)
    cmaes file and converts to a `pandas dataframe` and appends two columns to the dataframe,
    namely the number of restarts and the current population size.
    """

    def append_restart_column(init_df:pd.DataFrame)->pd.DataFrame:
        # Extract the iteration column
        init_df['restart_count'] = (init_df['iteration'] == 1).cumsum()-1

        return init_df
    
    def append_lambda_column(init_df:pd.DataFrame)->pd.DataFrame:
        init_df['lamda'] = init_df.groupby('restart_count')['evaluation'].diff()
        init_df['lamda'] = init_df['lamda'].fillna(init_df.groupby('restart_count')['lamda'].shift(-1))

        init_df['lamda'] = init_df['lamda'].fillna(init_df['lamda'].shift(1) * 2)

        init_df['lamda'] =init_df['lamda'].astype(int)
        return init_df

    def load_function(data_path:Union[str,Path])->pd.DataFrame:
        cur_df = outcmaes_func(data_path)
        cur_df_1 = append_restart_column(cur_df)
        cur_df_2 = append_lambda_column(cur_df_1)

        return cur_df_2

    return load_function

@append_global_iteration_lambda_outcmaes
def load_fit_cmaes(path:Union[str,Path])->pd.DataFrame:
    r"""
    This is just a function which calls the `fit.dat` generated from a CMA-ES run
    by using `pycmaes` (The library from Niko Hansen)
    """

    if not path.endswith("fit.dat"):
        raise AttributeError("The path does not correspond to the `fit.dat` file",
                             name="path",
                             obj=path)
    
    # These are the tile columns
    # This file does not require computing some dimension of the problem
    TITLE_COLUMNS:dict = {"iteration":int, 
                          "evaluation":int, 
                          "sigma":float,
                          "axis_ratio":float, 
                          "bestever":float, 
                          "best":float, 
                          "median":float, 
                          "worst_objective_function_value":float}
    
    # Generate the `pandas` Dataframe
    import warnings

    warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)
    with warnings.catch_warnings():
        df = pd.read_csv(path,
                        sep=" ",
                        header=None,
                        skiprows=[0],
                        names=TITLE_COLUMNS.keys(),
                        index_col=False,
                        dtype=TITLE_COLUMNS,
                        )
    
    return df

@append_global_iteration_lambda_outcmaes
def load_xmean_cmaes(path:Union[str,Path])->pd.DataFrame:
    r"""
    This is just a function which calls the `xmean.dat` generated from a CMA-ES run
    by using `pycmaes` (The library from Niko Hansen)
    """

    if not path.endswith("xmean.dat"):
        raise AttributeError("The path does not correspond to the `xmean.dat` file",
                             name="path",
                             obj=path)
    
    # These are the tile columns
    # NOTE:This file does require computing some dimension of the problem
    
    TITLE_COLUMNS:dict = {"iteration":int, 
                          "evaluation":int, 
                          "void0":int, 
                          "void1":int, 
                          "void2":int, 
                          "xmean":object}
    
    with open(file=path,mode="r",encoding="utf-8") as f:
        _ = f.readline()
        second_line = f.readline()

        # Get the number of variables by performing the split
        items = second_line.split(" ")

        #close the IOStream
        f.close()
    
    
    ncols_in_file:int = len(items)
    problem_dimension = ncols_in_file - len(TITLE_COLUMNS) + 1

    del items, second_line
    
    # Extract the void columns
    void_cols = []
    non_void_idxs = []

    for idx, title in enumerate(TITLE_COLUMNS.keys()):
        if title.startswith("void"):
            void_cols.append(title)
        else:
            non_void_idxs.append(idx)

    for idx in range(6,5+problem_dimension):
        non_void_idxs.append(idx)
    
    # Generate the `pandas` Dataframe
    df = pd.read_csv(filepath_or_buffer=path,
                     sep=" ",
                     skiprows=0,
                     #header=0,
                     #names=TITLE_COLUMNS.keys(),
                     index_col=False,
                     usecols=non_void_idxs,
                     #dtype=TITLE_COLUMNS,
                     )
    
    for a in range(2):
        cur_col_name = df.columns[a]

        # Reassign the name
        curkey = [*TITLE_COLUMNS][a]
        df = df.rename(columns={cur_col_name:curkey})
    
    for a in range(2,(2+problem_dimension)):
        cur_col_name = df.columns[a]
        df = df.rename(columns={cur_col_name:f"x{a-2}"})
    

    del a
    
    return df

@append_global_iteration_lambda_outcmaes
def load_xrecentbest_cmaes(path:Union[str,Path])->pd.DataFrame:
    r"""
    This is just a function which calls the `xrecentbest.dat` generated from a CMA-ES run
    by using `pycmaes` (The library from Niko Hansen)
    """

    if not path.endswith("xrecentbest.dat"):
        raise AttributeError("The path does not correspond to the `xrecentbest.dat` file",
                             name="path",
                             obj=path)
    
    # These are the tile columns
    # NOTE:This file does require computing some dimension of the problem
    
    TITLE_COLUMNS:dict = {"iteration":int, 
                          "evaluation":int, 
                          "sigma":int, 
                          "0":int, 
                          "fitness":int, 
                          "xbest":object}
    
    
    with open(file=path,mode="r",encoding="utf-8") as f:
        _ = f.readline()
        second_line = f.readline()

        # Get the number of variables by performing the split
        items = second_line.split(" ")

        #close the IOStream
        f.close()
    
    
    ncols_in_file:int = len(items)
    problem_dimension = ncols_in_file - len(TITLE_COLUMNS) + 1

    del items, second_line
    
    # Extract the void columns
    void_cols = []
    non_void_idxs = []

    for idx, title in enumerate(TITLE_COLUMNS.keys()):
        if title.startswith("0"):
            void_cols.append(title)
        else:
            non_void_idxs.append(idx)

    for idx in range(6,5+problem_dimension):
        non_void_idxs.append(idx)
    
    # Generate the `pandas` Dataframe
    df = pd.read_csv(filepath_or_buffer=path,
                     sep=" ",
                     skiprows=0,
                     #header=0,
                     #names=TITLE_COLUMNS.keys(),
                     index_col=False,
                     usecols=non_void_idxs,
                     #dtype=TITLE_COLUMNS,
                     )
    
    for a in range(4):
        cur_col_name = df.columns[a]

        # Reassign the name
        curkey = [*TITLE_COLUMNS][non_void_idxs[a]]
        df = df.rename(columns={cur_col_name:curkey})
    
    for a in range(4,(4+problem_dimension)):
        cur_col_name = df.columns[a]
        df = df.rename(columns={cur_col_name:f"x{a-4}"})
    

    del a
    
    return df

def load_cmaes_file_as_dataframe(path:Union[str,Path])->pd.DataFrame:
    
    if path.endswith("fit.dat"):
        df = load_fit_cmaes(path)
    elif path.endswith("xmean.dat"):
        df = load_xmean_cmaes(path)
    elif path.endswith("xrecentbest.dat"):
        df = load_xrecentbest_cmaes(path)
    else:
        raise AttributeError("The function is not implemented for other type of file")
    
    
    return df

def load_particular_data(foldername:Union[str,Path])->dict:
    r"""
    This function receives a path of a folder and returns a list comprised of loaded dataframes with
    information about a particular run identified by the folder

    Args:
    -------------------
    - foldername: `Union[str,Path]`: A path with the information of a run

    Returns:
    ------------------
    - A list of Dataframes with the information of the run
    """

    foldername = _convert_2_path_object(foldername)
    
    IOH_file:str = None

    file_struct = os.listdir(foldername.absolute())
    
    # Get the IOH Profiler File within
    for filename in file_struct:
        if filename.endswith(".json"):
            # Assign the filename
            IOH_file = os.path.join(foldername.absolute(),filename)

            # Break the loop
            break
    
    del filename

    cmaes_path = os.path.join(foldername.absolute(),"outcmaes")
    cmaes_files = os.listdir(cmaes_path)


    if not os.path.exists(cmaes_path):
        raise FileNotFoundError()
    
    # Generate the IOH object
    ioh_obj:IOH_Parser = IOH_Parser(IOH_file)

    #Generate a names array
    names_arr:list = ["evaluations"]
    
    # Generate a list that stores dataframes
    dataframes_arr:List[pd.DataFrame] = [ioh_obj.return_complete_table_per_instance(0)]

    for cmaes_file in cmaes_files:
        # Loop all over the files
        if cmaes_file in RELEVANT_OUTCMAES:
            dataframes_arr.append(load_cmaes_file_as_dataframe(os.path.join(cmaes_path,cmaes_file)))
            names_arr.append(cmaes_file.removesuffix(".dat"))


    return_dict = {name_i: dataf_i for name_i,dataf_i in list(zip(names_arr,dataframes_arr))}

    return return_dict

def plot_run_iterations_on_contour(trial_path:Union[str,Path],df_dict:dict)->None:
    r"""
    This function is meant to plot the results of each iteration of a run.

    Args:
    ---------------
    - trial_path: `Union[str,Path]`: Path indicating the repository of lookup folder of the run
    - df_dict: `dict` : A dictionary pointing to the dataframes with relevant data of corresponding run
    """

    import matplotlib.pyplot as plt

    # Break down the path to extract information regarding the function
    data_dict:dict = _return_identifiers_from_file_path(trial_path)

    # Get the function type, invariance mode, function id and run
    func_type:str = data_dict['func_type']
    inv_mode:str = data_dict['inv_mode']
    cur_alg:str = data_dict['cur_alg']
    func_id:int = data_dict['func_id']
    func_name:str = data_dict['func_name']
    run_:int = data_dict['run']

    if not os.path.exists(os.path.join(os.getcwd(),"Plots_Definite",func_type,inv_mode,cur_alg,f"f{func_id}",f"Run_{run_}")):
           # Generate a path
           path_:Path = Path(os.path.join(os.getcwd(),"Plots_Definite",func_type,inv_mode,cur_alg,f"f{func_id}",f"Run_{run_}"))
           path_.mkdir(parents=True)

    # Find the main folder data of two-dimensional problems
    data_2D_prob:str = os.path.join(ROOT_TWO_DIMENSIONAL_PROBLEMS_REPO,func_type,inv_mode,f"f{func_id}")

    # load the data to generate the contour
    data_file_path:str = os.path.join(data_2D_prob,"data.npy")
    with open(data_file_path,'rb') as ff:
        xx = np.load(ff)
        yy = np.load(ff)
        zz = np.load(ff)

    # plot the contour first
    fig1, ax2 = plt.subplots(layout='constrained')
    CS = ax2.contourf(xx, yy, zz, 30, cmap=plt.cm.coolwarm)
    fig1.colorbar(CS, ax=ax2, shrink=0.9)


    # Plot the labels
    ax2.set_xlabel(f"$x_{0}$")
    ax2.set_ylabel(f"$x_{1}$")
    #ax2.set_title(f"Function: {func_name}, mode: {inv_mode} \n Run: {run_}")
    #ax2.legend(loc="best")

    # Perform a loop all over the fit.dat dataset
    fit_dataset:pd.DataFrame = df_dict['fit']
    evaluations_dataset:pd.DataFrame = df_dict['evaluations']

    cur_pts, = ax2.plot([0,1],[0,1],marker="D", color="black", markerfacecolor="black" ,linestyle='None',)
    #plt.ion()
    for row_t in fit_dataset.itertuples():
       # Extract the current iteration, evaluation and lambda
       cur_iter = int(row_t.Index + 1)
       cur_max_evaluation = int(row_t.evaluation)
       cur_lambda = int(row_t.lamda)

       init_eval_idx = int(cur_max_evaluation-cur_lambda+1)
       final_eval_idx = int(cur_max_evaluation)

       # Slice the evaluations dataset with the evaluations indices
       relevant_evaluations = evaluations_dataset[(evaluations_dataset['evaluations'] >= init_eval_idx) & (evaluations_dataset['evaluations'] <= final_eval_idx)]

       cur_x0, cur_x1 = (relevant_evaluations['x0'].to_numpy(),
                         relevant_evaluations['x1'].to_numpy())
       ax2.set_title(f"Iteration: {cur_iter} lambda: {cur_lambda} ")
       #current_points = ax2.scatter(x=cur_x0,y=cur_x1,marker=".", color="black")
       cur_pts.set_xdata(cur_x0)
       cur_pts.set_ydata(cur_x1)
       fig1.canvas.draw()

       

       plt.savefig(os.path.join(os.getcwd(),"Plots_Definite",func_type,inv_mode,cur_alg,f"f{func_id}",f"Run_{run_}",
                                f'plot_iteration_{cur_iter}.pdf'),format="pdf",dpi=100)
       #fig1.canvas.flush_events()
    
   
def load_experiment_file(exp_file:Union[str,Path])->pd.DataFrame:
    r"""
    This function is meant to just load the experiment setup file into memory

    Args:
    -------------------
    exp_file: `Union[str,Path]`: This is a path indicating the route to load the experiment setup file
    """

    # Convert the file to path
    path_:Path = _convert_2_path_object(exp_file)

    if path_.exists() and path_.suffix in ([".xlsx",".ods"]):
        df:pd.DataFrame = pd.read_excel(path_.absolute(),header=0)
    elif path_.exists() and path_.suffix in ([".txt",".dat",".csv"]):
        df:pd.DataFrame = pd.read_csv(path_.absolute(),header=0,sep=",")
    else:
        raise ValueError("The file is not an Excel Workbook or a text datasheet")
    
    return df


def _classify_run_folders(exp_file: Union[str, Path], 
                         root_experiment_repo: Union[str, Path], 
                         mode: int = 1) -> Union[List[List[str]],
                                                 List[List[List[str]]],
                                                 List[List[List[List[str]]]]
                         ]:
    r"""
    This function performs classification of files based on a specific mode in order to generate corresponding results.

    Args:
    ----------
    - exp_file: `Union[str, Path]`: Path or string to the experiment file.
    - root_experiment_repo: `Union[str, Path]`: Path or string to the root directory of the experiment repository.
    - mode: `int`: Integer specifying the mode of classification (default is 1).

    Returns:
    ----------
    - A list of lists, where each sublist contains classified files or directories based on the specified mode.

    Raises:
    ----------
    - AttributeError: If the given root_experiment_repo path does not exist.
    """

    from itertools import product

    def mode_1(path_obj:Path,
               exp_df:pd.DataFrame)->List[List[str]]:
        r"""
        This is the definition of the classification mode 1;
        The lists will consist of having type/invariance_mode/function/algorithm/groups of 4 sigmas 
        (same seed)


        Args:
        -------------------
        - path_obj: `Path`: A path object pointing to the root directory of the dataset.
        - exp_df: `pd.Dataframe`: A dataframe with information of properties of the run.
        """

        iterable_of_function_algorithm = list(product(PROBLEMS_MODULES,INVARIANCE_MODES,ALGORITHMS_MODULES,FID_LIMITS))
        unique_seeds = pd.unique(exp_df['Seed'])

        list_of_lists:list = []

        for prob_mod_i,inv_mod_i, alg_mod_i, fid_i in iterable_of_function_algorithm:

            # Iterate over the same seed runs
            for seed in unique_seeds:
                # Get the runs with 'iso-seed'
                id_df:pd.DataFrame = exp_df[exp_df["Seed"]==seed]
                runs_to_iterate = id_df['Run'].to_numpy(dtype=int)

                # Initialize the list
                partial_list:list = []

                for i_run in runs_to_iterate:
                    # Build-up the string to search
                    search_string:str = os.path.join(path_obj.absolute(),
                                                    prob_mod_i,
                                                    inv_mod_i,
                                                    alg_mod_i,
                                                    str(f"f{fid_i}"),f"Run_{i_run}").replace("\\",'/')

                    # Look for matching substrings
                    partial_list.append(search_string)
                
                # Append the list
                list_of_lists.append(partial_list)
        

        

        return list_of_lists
    
    def mode_2(path_obj:Path,
               exp_df:pd.DataFrame)->List[List[str]]:
        r"""
        This is the definition of the classification mode 2;
        The lists will consist of having type/invariance_mode/function/algorithm/groups of 30 runs
        (same seed)


        Args:
        -------------------
        - path_obj: `Path`: A path object pointing to the root directory of the dataset.
        - exp_df: `pd.Dataframe`: A dataframe with information of properties of the run.
        """

        iterable_of_function_algorithm = list(product(PROBLEMS_MODULES,
                                                      INVARIANCE_MODES,
                                                      ALGORITHMS_MODULES,
                                                      FID_LIMITS))
        unique_sigmas = pd.unique(exp_df['Step_Size'])

        list_of_lists:list = []

        for prob_mod_i,inv_mod_i, alg_mod_i, fid_i in iterable_of_function_algorithm:

            # Iterate over the same seed runs
            for sigma in unique_sigmas:
                # Get the runs with 'iso-sigma'
                id_df:pd.DataFrame = exp_df[exp_df['Step_Size']==sigma]
                runs_to_iterate = id_df['Run'].to_numpy(dtype=int)

                # Initialize the list
                partial_list:list = []

                for i_run in runs_to_iterate:
                    # Build-up the string to search
                    search_string:str = os.path.join(path_obj.absolute(),
                                                    prob_mod_i,
                                                    inv_mod_i,
                                                    alg_mod_i,
                                                    str(f"f{fid_i}"),f"Run_{i_run}").replace("\\",'/')

                    # Look for matching substrings
                    partial_list.append(search_string)
                
                # Append the list
                list_of_lists.append(partial_list)


        return list_of_lists


    def mode_3(path_obj:Path,
               exp_df:pd.DataFrame)->List[List[List[str]]]:
        r"""
        This is the definition of the classification mode 3;
        The lists will consist of having type/invariance_mode/function/algorithm/groups of 120 runs


        Args:
        -------------------
        - path_obj: `Path`: A path object pointing to the root directory of the dataset.
        - exp_df: `pd.Dataframe`: A dataframe with information of properties of the run.
        """

        iterable_of_function_algorithm = list(product(PROBLEMS_MODULES,
                                                      INVARIANCE_MODES,
                                                      ALGORITHMS_MODULES,
                                                      FID_LIMITS))
        unique_sigmas = pd.unique(exp_df['Step_Size'])

        list_of_lists:list = []

        for prob_mod_i,inv_mod_i, alg_mod_i, fid_i in iterable_of_function_algorithm:

            # Initialize an empty list to store the iso-sigma_runs
            partial_high_list:list = []

            # Iterate over the same seed runs
            for sigma in unique_sigmas:
                # Get the runs with 'iso-sigma'
                id_df:pd.DataFrame = exp_df[exp_df['Step_Size']==sigma]
                runs_to_iterate = id_df['Run'].to_numpy(dtype=int)

                # Initialize the list
                partial_list:list = []

                for i_run in runs_to_iterate:
                    # Build-up the string to search
                    search_string:str = os.path.join(path_obj.absolute(),
                                                    prob_mod_i,
                                                    inv_mod_i,
                                                    alg_mod_i,
                                                    str(f"f{fid_i}"),f"Run_{i_run}").replace("\\",'/')

                    # Look for matching substrings
                    partial_list.append(search_string)
                
                partial_high_list.append(partial_list)
                # Append the list
            list_of_lists.append(partial_high_list)
        
        return list_of_lists
    
    def mode_4(path_obj:Path,
               exp_df:pd.DataFrame)->List[List[str]]:
        r"""
        This is the definition of the classification mode 4;
        The lists will consist of having type/invariance_mode/function base,

        Then the sublists will contain the pairs algorithm/run


        Args:
        -------------------
        - path_obj: `Path`: A path object pointing to the root directory of the dataset.
        - exp_df: `pd.Dataframe`: A dataframe with information of properties of the run.
        """

        iterable_of_function = list(product(PROBLEMS_MODULES,
                                                      INVARIANCE_MODES,
                                                      FID_LIMITS))
        all_runs = pd.unique(exp_df['Run'])

        list_of_lists:list = []

        for prob_mod_i,inv_mod_i, fid_i in iterable_of_function:

            # Iterate over the same seed runs
            for i_run in all_runs:
                
                # Initialize the list
                partial_list = []
                # Loop all over the algorithms
                for alg_mod_i in ALGORITHMS_MODULES:


                    # Build-up the string to search
                    search_string:str = os.path.join(path_obj.absolute(),
                                                    prob_mod_i,
                                                    inv_mod_i,
                                                    alg_mod_i,
                                                    str(f"f{fid_i}"),f"Run_{i_run}").replace("\\",'/')

                    # Look for matching substrings
                    partial_list.append(search_string)
                
                # Append the list
                list_of_lists.append(partial_list)
        
        return list_of_lists
    
    def mode_5(path_obj:Path,
               exp_df:pd.DataFrame)->List[List[List[List[str]]]]:
        r"""
        This is the definition of the classification mode 5;
        The lists will consist of having type/invariance_mode/function base,

        Then the sublists will contain the pairs algorithm and then contain all the runs.


        Args:
        -------------------
        - path_obj: `Path`: A path object pointing to the root directory of the dataset.
        - exp_df: `pd.Dataframe`: A dataframe with information of properties of the run.
        """

        iterable_of_function = list(product(PROBLEMS_MODULES,
                                                      INVARIANCE_MODES,
                                                      FID_LIMITS))
        unique_sigma = pd.unique(exp_df['Step_Size'])

        list_of_lists:list = []

        for prob_mod_i,inv_mod_i, fid_i in iterable_of_function:
            
            # Initialize an empty list to store the iso-sigma_runs
            partial_high_list:list = []
            # Iterate over the same seed runs
            for i_sigma in unique_sigma:

                sigma_list:list = []
                
                partial_df = exp_df[exp_df["Step_Size"]==i_sigma]
                list_runs:pd.DataFrame = partial_df['Run']
                # Loop all over the algorithms
                for alg_mod_i in ALGORITHMS_MODULES:
                    
                    # Initialize the list
                    partial_list = []

                    # Loop all over the runs

                    for i_run in list_runs.to_numpy(dtype=int):
                        # Build-up the string to search
                        search_string:str = os.path.join(path_obj.absolute(),
                                                        prob_mod_i,
                                                        inv_mod_i,
                                                        alg_mod_i,
                                                        str(f"f{fid_i}"),f"Run_{i_run}").replace("\\",'/')

                        # Look for matching substrings
                        partial_list.append(search_string)

                    sigma_list.append(partial_list)
                partial_high_list.append(sigma_list)
                
            # Append the list
            list_of_lists.append(partial_high_list)
        
        return list_of_lists


    exp_file_df:pd.DataFrame = load_experiment_file(exp_file)
    root_experiment_repo_: Path = _convert_2_path_object(root_experiment_repo)

    # Check if the root experiment repository exists
    if  not root_experiment_repo_.exists():
        raise AttributeError("The given path does not exist",
                             name="root_experiment_repo",
                             obj=root_experiment_repo)

    #for ii, ff in enumerate(list_generator):
        #rint(ii, ff)
    #    a = 1

    if mode==1:
        return mode_1(root_experiment_repo_,
                      exp_file_df)
    elif mode==2:
        return mode_2(root_experiment_repo_,
                      exp_file_df)
    elif mode==3:
        return mode_3(root_experiment_repo_,
                      exp_file_df)

    elif mode==4:
        return mode_4(root_experiment_repo_,
                      exp_file_df)
    
    elif mode==5:
        return mode_5(root_experiment_repo_,
                      exp_file_df)



def plot_mode_1(exp_file: Union[str, Path], 
                root_experiment_repo: Union[str, Path],
                save_suffix:str = os.path.join(os.getcwd(),"plots","Mode_1"))->None:
    
    # Import matplotlib
    import matplotlib.pyplot as plt
    

    # Get the list of arrays with all the possible plots
    list_:List[List[str]] = _classify_run_folders(exp_file=exp_file,
                         root_experiment_repo=root_experiment_repo,
                         mode=1)
    
    # Load again the experiment file
    exp_file_df:pd.DataFrame = load_experiment_file(exp_file)
    
    # Start looping all over the list
    for idx_plot,sublist in enumerate(list_):
        # Initialize a plot object
        seed:Union[None,int] = None
        fig1, ax1 = plt.subplots(layout='constrained')

        ax1.set_xlabel("Evaluations")
        ax1.set_ylabel("Best so far")
        #ax1.set_yscale('log')
        for idx_curve,iFile in enumerate(sublist):

            # Break down the path to extract information regarding the function
            data_dict:dict = _return_identifiers_from_file_path(iFile)

            # Load the data
            df_data:dict = load_particular_data(iFile)

            # Extract the evaluations dataset
            evaluations_dataset:pd.DataFrame = df_data['evaluations']
            
            # Get the function type, invariance mode, function id and run
            func_type:str = data_dict['func_type']
            inv_mode:str = data_dict['inv_mode']
            cur_alg:str = data_dict['cur_alg']
            func_id:int = data_dict['func_id']
            func_name:str = data_dict['func_name']
            run_:int = data_dict['run']
            
            # Get the sigma/step size
            cur_step_size:float = exp_file_df[exp_file_df["Run"]==run_].loc[:,"Step_Size"].to_numpy().ravel()[0]

            if idx_curve == 0:
                # Match the run with the seed
                seed = exp_file_df[exp_file_df["Run"]==run_].loc[:,"Seed"].to_numpy().ravel()[0]

            # plot the curves
            ax1.plot(evaluations_dataset.loc[:,["evaluations"]].to_numpy(),
                     evaluations_dataset.loc[:,["raw_y_best"]].to_numpy(),
                     label=f"$\sigma_0={cur_step_size}$")
            
        
        ax1.set_title(f"Function: {func_name}, Seed: {str(seed)}")
        ax1.legend(loc='best')
        
        save_path:Path = _convert_2_path_object(os.path.join(save_suffix,
                                                            func_type,
                                                            inv_mode,
                                                            cur_alg,
                                                            f"f{func_id}",
                                                            str(seed)))
        # Generate a path object
        if not save_path.exists():
            save_path.mkdir(exist_ok=False,parents=True)
            

        plt.savefig(os.path.join(save_path.absolute(),"convergence.pdf"),format="pdf")

        # Close the figure
        plt.close(fig1)

def plot_mode_2(exp_file: Union[str, Path], 
                root_experiment_repo: Union[str, Path],
                save_suffix:str = os.path.join(os.getcwd(),"plots","Mode_2"))->None:
    
    # Import matplotlib
    import matplotlib.pyplot as plt
    

    # Get the list of arrays with all the possible plots
    list_:List[List[str]] = _classify_run_folders(exp_file=exp_file,
                         root_experiment_repo=root_experiment_repo,
                         mode=2)
    
    # Load again the experiment file
    exp_file_df:pd.DataFrame = load_experiment_file(exp_file)
    
    # Start looping all over the list
    for idx_plot,sublist in enumerate(list_):
        # Initialize a plot object
        fig1, ax1 = plt.subplots(layout='constrained')

        ax1.set_xlabel("Evaluations")
        ax1.set_ylabel("Best so far")

        #ax1.set_yscale('log')
        for idx_curve,iFile in enumerate(sublist):

            # Break down the path to extract information regarding the function
            data_dict:dict = _return_identifiers_from_file_path(iFile)

            # Load the data
            df_data:dict = load_particular_data(iFile)

            # Extract the evaluations dataset
            evaluations_dataset:pd.DataFrame = df_data['evaluations']
            
            # Get the function type, invariance mode, function id and run
            func_type:str = data_dict['func_type']
            inv_mode:str = data_dict['inv_mode']
            cur_alg:str = data_dict['cur_alg']
            func_id:int = data_dict['func_id']
            func_name:str = data_dict['func_name']
            run_:int = data_dict['run']
            

            if idx_curve == 0:
                # Match the run with the step size
                cur_step_size:float = exp_file_df[exp_file_df["Run"]==run_].loc[:,"Step_Size"].to_numpy().ravel()[0]
                step_size_index:int = exp_file_df[exp_file_df["Run"]==run_].index.to_numpy().ravel()[0] + 1

            # plot the curves
            ax1.plot(evaluations_dataset.loc[:,["evaluations"]].to_numpy(),
                     evaluations_dataset.loc[:,["raw_y_best"]].to_numpy(),
                     color='black')
            
        
        ax1.set_title(f"Function: {func_name}, Sigma: {str(cur_step_size)}, Runs Number: {len(sublist)}")
        
        save_path:Path = _convert_2_path_object(os.path.join(save_suffix,
                                                            func_type,
                                                            inv_mode,
                                                            cur_alg,
                                                            f"f{func_id}",
                                                            f"Step_Size_{step_size_index}"))
        # Generate a path object
        if not save_path.exists():
            save_path.mkdir(exist_ok=False,parents=True)
            

        plt.savefig(os.path.join(save_path.absolute(),"convergence.pdf"),format="pdf")

        # Close the figure
        plt.close(fig1)


def plot_mode_3(exp_file: Union[str, Path], 
                root_experiment_repo: Union[str, Path],
                save_suffix:str = os.path.join(os.getcwd(),"plots","Mode_3"))->None:
    
    # Import matplotlib
    import matplotlib.pyplot as plt
    

    # Get the list of arrays with all the possible plots
    list_:List[List[str]] = _classify_run_folders(exp_file=exp_file,
                         root_experiment_repo=root_experiment_repo,
                         mode=3)
    
    # Load again the experiment file
    exp_file_df:pd.DataFrame = load_experiment_file(exp_file)
    
    # Start looping all over the list
    for idx_plot,sublist in enumerate(list_):
        
        # Initialize a plot object
        fig1, ax1 = plt.subplots(layout='constrained')

        ax1.set_xlabel("Evaluations")
        ax1.set_ylabel("Best so far")

        #ax1.set_yscale('log')
        for idx_sigma,i_list_sigma in enumerate(sublist):

            # Compressed Evaluations df 
            df_compressed:Union[None,pd.DataFrame] = None # Initialize
            
            for idx_file, iFile in enumerate(i_list_sigma):

                # Break down the path to extract information regarding the function
                data_dict:dict = _return_identifiers_from_file_path(iFile)
                # Load the data
                df_data:dict = load_particular_data(iFile)

                # Extract the evaluations dataset
                evaluations_dataset:pd.DataFrame = df_data['evaluations']
                
                # Get the function type, invariance mode, function id and run
                func_type:str = data_dict['func_type']
                inv_mode:str = data_dict['inv_mode']
                cur_alg:str = data_dict['cur_alg']
                func_id:int = data_dict['func_id']
                func_name:str = data_dict['func_name']
                run_:int = data_dict['run']

                if idx_file == 0:
                    df_compressed = evaluations_dataset.copy()
                    # Reassign the runs column
                    df_compressed["run"] = df_compressed["run"].replace(1, run_)
                else:
                    # Change the run number
                    evaluations_dataset["run"] = evaluations_dataset["run"].replace(1, run_)

                    # append to last
                    df_compressed = pd.concat([df_compressed, evaluations_dataset],axis=0)
                
                

                # if idx_curve == 0:
                #     # Match the run with the step size
                #     cur_step_size:float = exp_file_df[exp_file_df["Run"]==run_].loc[:,"Step_Size"].to_numpy().ravel()[0]
                #     step_size_index:int = exp_file_df[exp_file_df["Run"]==run_].index.to_numpy().ravel()[0] + 1

            # Get the current sigma
            cur_sigma:float = exp_file_df.loc[idx_sigma,"Step_Size"]

            # Get a consolidated average and standard deviation
            mean_df, std_df, count_df = (df_compressed.groupby(['evaluations']).mean(),
                                         df_compressed.groupby(['evaluations']).std(),
                                         df_compressed.groupby(['evaluations']).count())

            r"""
            mean_arr,std_arr, count_arr = (data_array[idx].groupby(['evaluations']).mean(), 
                                       data_array[idx].groupby(['evaluations']).std(), 
                                       data_array[idx].groupby(['evaluations']).count())
        
                    ax_ptr.plot(mean_arr.index.to_numpy(),-1*mean_arr['Objective'],label=f"{DIMENSIONS[idx]}D")
                    up_bound = -1*mean_arr['Objective'].to_numpy() + 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
                    lo_bound = -1*mean_arr['Objective'].to_numpy() - 1.96*std_arr['Objective'].to_numpy()/np.sqrt(count_arr['Objective'].to_numpy())
                    ax_ptr.fill_between(x=mean_arr.index.to_numpy(),y1 =  up_bound, y2 = lo_bound, alpha=0.2)
            
            """

            # plot the curves
            ax1.plot(mean_df.index.to_numpy(),
                       mean_df['raw_y_best'].to_numpy(),
                        label=f"$\sigma_0=${cur_sigma}")
            
            up_bound = mean_df['raw_y_best'].to_numpy() +  1.96*std_df['raw_y_best'].to_numpy()/np.sqrt(count_df['raw_y_best'].to_numpy())
            lo_bound = mean_df['raw_y_best'].to_numpy() -  1.96*std_df['raw_y_best'].to_numpy()/np.sqrt(count_df['raw_y_best'].to_numpy())

            ax1.fill_between(x=mean_df.index.to_numpy(),
                             y1= up_bound,
                             y2= lo_bound,
                             alpha=0.2)
            
        
            ax1.set_title(f"Function: {func_name}, Algorithm: {cur_alg}")
        
        ax1.legend(loc="best")
        save_path:Path = _convert_2_path_object(os.path.join(save_suffix,
                                                            func_type,
                                                            inv_mode,
                                                            cur_alg,
                                                            f"f{func_id}",
                                                            ))
        # Generate a path object
        if not save_path.exists():
            save_path.mkdir(exist_ok=False,parents=True)
                

        plt.savefig(os.path.join(save_path.absolute(),"convergence.pdf"),format="pdf")

            # Close the figure
        plt.close(fig1)

def plot_mode_4(exp_file: Union[str, Path], 
                root_experiment_repo: Union[str, Path],
                save_suffix:str = os.path.join(os.getcwd(),"plots","Mode_4"))->None:
    
    # Import matplotlib
    import matplotlib.pyplot as plt
    

    # Get the list of arrays with all the possible plots
    list_:List[List[str]] = _classify_run_folders(exp_file=exp_file,
                         root_experiment_repo=root_experiment_repo,
                         mode=4)
    
    # Load again the experiment file
    exp_file_df:pd.DataFrame = load_experiment_file(exp_file)
    
    # Start looping all over the list
    for idx_plot,sublist in enumerate(list_):
        
        # Initialize a plot object
        fig1, ax1 = plt.subplots(layout='constrained')

        ax1.set_xlabel("Evaluations")
        ax1.set_ylabel("Best so far")

        actual_seed = None
            
        for idx_file, iFile in enumerate(sublist):

            # Break down the path to extract information regarding the function
            data_dict:dict = _return_identifiers_from_file_path(iFile)
            # Load the data
            df_data:dict = load_particular_data(iFile)

            # Extract the evaluations dataset
            evaluations_dataset:pd.DataFrame = df_data['evaluations']
            
            # Get the function type, invariance mode, function id and run
            func_type:str = data_dict['func_type']
            inv_mode:str = data_dict['inv_mode']
            cur_alg:str = data_dict['cur_alg']
            func_id:int = data_dict['func_id']
            func_name:str = data_dict['func_name']
            run_:int = data_dict['run']


            if idx_file == 0:
                # Match the run with the step size
                cur_step_size:float = exp_file_df[exp_file_df["Run"]==run_].loc[:,"Step_Size"].to_numpy().ravel()[0]
                step_size_index:int = exp_file_df[exp_file_df["Run"]==run_].index.to_numpy().ravel()[0] + 1
                actual_seed = exp_file_df[exp_file_df["Run"]==run_].loc[:,"Seed"].to_numpy().ravel()[0]

            # plot the curves
            ax1.plot(evaluations_dataset.loc[:,["evaluations"]].to_numpy(),
                     evaluations_dataset.loc[:,["raw_y_best"]].to_numpy(),
                     label=cur_alg)
            
        
            ax1.set_title(f"Function: {func_name}, Seed:{actual_seed}, $\sigma_0=${cur_step_size}")
        
        ax1.legend(loc="best")
        save_path:Path = _convert_2_path_object(os.path.join(save_suffix,
                                                            func_type,
                                                            inv_mode,
                                                            f"f{func_id}",
                                                            f"Run_{run_}"
                                                            ))
        # Generate a path object
        if not save_path.exists():
            save_path.mkdir(exist_ok=False,parents=True)
                

        plt.savefig(os.path.join(save_path.absolute(),"convergence.pdf"),format="pdf")

            # Close the figure
        plt.close(fig1)

def plot_mode_5(exp_file: Union[str, Path], 
                root_experiment_repo: Union[str, Path],
                save_suffix:str = os.path.join(os.getcwd(),"plots","Mode_5"))->None:
    
    # Import matplotlib
    import matplotlib.pyplot as plt
    

    # Get the list of arrays with all the possible plots
    list_:List[List[str]] = _classify_run_folders(exp_file=exp_file,
                         root_experiment_repo=root_experiment_repo,
                         mode=5)
    
    # Load again the experiment file
    exp_file_df:pd.DataFrame = load_experiment_file(exp_file)

    all_sigmas = pd.unique(exp_file_df['Step_Size'])
    
    # Start looping all over the list
    for idx_plot,sublist in enumerate(list_):
        

        #ax1.set_yscale('log')
        for idx_sigma,i_list_sigma in enumerate(sublist):

            # Initialize a plot object
            fig1, ax1 = plt.subplots(layout='constrained')

            ax1.set_xlabel("Evaluations")
            ax1.set_ylabel("Best so far")
            
            for idx_alg, i_list_alg in enumerate(i_list_sigma):
                    
                # Compressed Evaluations df 
                df_compressed:Union[None,pd.DataFrame] = None # Initialize
                    
                for idx_file, iFile in enumerate(i_list_alg):

                    # Break down the path to extract information regarding the function
                    data_dict:dict = _return_identifiers_from_file_path(iFile)
                    # Load the data
                    df_data:dict = load_particular_data(iFile)

                    # Extract the evaluations dataset
                    evaluations_dataset:pd.DataFrame = df_data['evaluations']
                    
                    # Get the function type, invariance mode, function id and run
                    func_type:str = data_dict['func_type']
                    inv_mode:str = data_dict['inv_mode']
                    cur_alg:str = data_dict['cur_alg']
                    func_id:int = data_dict['func_id']
                    func_name:str = data_dict['func_name']
                    run_:int = data_dict['run']

                    if idx_file == 0:
                        df_compressed = evaluations_dataset.copy()
                        # Reassign the runs column
                        df_compressed["run"] = df_compressed["run"].replace(1, run_)
                    else:
                        # Change the run number
                        evaluations_dataset["run"] = evaluations_dataset["run"].replace(1, run_)

                        # append to last
                        df_compressed = pd.concat([df_compressed, evaluations_dataset],axis=0)

                # Get the current sigma
                cur_sigma:float = all_sigmas[idx_sigma]

                # Get a consolidated average and standard deviation
                mean_df, std_df, count_df = (df_compressed.groupby(['evaluations']).mean(),
                                            df_compressed.groupby(['evaluations']).std(),
                                            df_compressed.groupby(['evaluations']).count())

                # plot the curves
                ax1.plot(mean_df.index.to_numpy(),
                        mean_df['raw_y_best'].to_numpy(),
                            label=cur_alg)
            
                up_bound = mean_df['raw_y_best'].to_numpy() +  1.96*std_df['raw_y_best'].to_numpy()/np.sqrt(count_df['raw_y_best'].to_numpy())
                lo_bound = mean_df['raw_y_best'].to_numpy() -  1.96*std_df['raw_y_best'].to_numpy()/np.sqrt(count_df['raw_y_best'].to_numpy())

                ax1.fill_between(x=mean_df.index.to_numpy(),
                                y1= up_bound,
                                y2= lo_bound,
                                alpha=0.2)
            
        
                ax1.set_title(f"Function: {func_name}, $\sigma_0=$ {cur_sigma}")
        
            ax1.legend(loc="best")
            save_path:Path = _convert_2_path_object(os.path.join(save_suffix,
                                                                func_type,
                                                                inv_mode,
                                                                f"f{func_id}",
                                                                f"Step_size_{idx_sigma+1}"
                                                                ))
            # Generate a path object
            if not save_path.exists():
                save_path.mkdir(exist_ok=False,parents=True)
                    

            plt.savefig(os.path.join(save_path.absolute(),"convergence.pdf"),format="pdf")

                # Close the figure
            plt.close(fig1)
#trial_path:str = "C:/Users/iolar/Documents/Modular_Problems_IOH/ModularExperiments/NonBBOB/average/injection/f10/Run_1"
#trial_path:str = "C:/Users/iolar/Documents/Modular_Problems_IOH/ModularExperiments/NonBBOB/average/rejection_sampling/f10/Run_4"
#trial_path:str = "C:/Users/iolar/Documents/Modular_Problems_IOH/ModularExperiments/NonBBOB/average/pure/f10/Run_1"
#trial_path:str = "C:/Users/iolar/Documents/Modular_Problems_IOH/ModularExperiments/NonBBOB/average/pure/f8/Run_6"
#trial_path:str = "C:/Users/iolar/Documents/Modular_Problems_IOH/ModularExperiments/NonBBOB/average/rejection_sampling/f8/Run_6"

#trial_path:str = "C:/Users/iolar/Documents/Modular_Problems_IOH/ModularExperiments/NonBBOBWithSubspace/average/rejection_sampling/f6/Run_6"
#trial_path:str = "C:/Users/iolar/Downloads/Loaded_ModularExperiments/NonBBOB/average/pure/f1/Run_100"
#trial_path:str = "C:/Users/iolar/Downloads/Loaded_ModularExperiments/NonBBOB/average/injectioN/f7/Run_1"

#list_ = load_particular_data(trial_path)
#print(list_['fit'])
#print(list_['evaluations'])

#plot_run_iterations_on_contour(trial_path,list_)

#resulting_list:list = classify_run_folders(
#                        exp_file="C:/Users/iolar/Documents/Modular_Problems_IOH/Dataset_1.csv",
##                        root_experiment_repo="C:/Users/iolar/Downloads/Loaded_ModularExperiments",
 #                       mode=1)

plot_mode_5("C:/Users/iolar/Documents/Modular_Problems_IOH/Dataset_1.csv",
            "C:/Users/iolar/Downloads/Loaded_ModularExperiments")



