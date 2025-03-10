import os
import time
from ioh import get_problem
import ioh
from typing import List, Optional, Union
from numpy import ndarray
from numpy import matrix
from numpy import ravel
import warnings
from typing import overload




class MyIOHFormatOnEveryEvaluationLogger:
    def __init__(self, folder_name='TMP', algorithm_name='UNKNOWN', suite='unknown suite', algorithm_info='algorithm_info'):
        self.folder_name = MyIOHFormatOnEveryEvaluationLogger.__generate_dir_name(folder_name)
        self.algorithm_name = algorithm_name
        self.algorithm_info = algorithm_info
        self.suite = suite
        self.create_time = time.process_time()

    @staticmethod
    def __generate_dir_name(name, x=0):
        while True:
            dir_name = (name + ('-' + str(x))).strip()
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
                return dir_name
            else:
                x = x + 1

    def watch(self, algorithm, extra_data):
        self.algorithm = algorithm
        self.extra_info_getters = extra_data

    def _set_up_logger(self, fid, iid, dim, func_name):
        self.log_info_path = f'{self.folder_name}/IOHprofiler_f{fid}_{func_name}.info'
        with open(self.log_info_path, 'a') as f:
            f.write(f'suite = \"{self.suite}\", funcId = {fid}, funcName = \"{func_name}\", DIM = {dim}, maximization = \"F\", algId = \"{self.algorithm_name}\", algInfo = \"{self.algorithm_info}\"\n')
        self.log_file_path = f'data_f{fid}_{func_name}/IOHprofiler_f{fid}_DIM{dim}.dat'
        self.log_file_full_path = f'{self.folder_name}/{self.log_file_path}'
        os.makedirs(os.path.dirname(self.log_file_full_path), exist_ok=True)
        self.first_line = 0
        self.last_line = 0
        with open(self.log_file_full_path, 'a') as f:
            f.write('\"function evaluation\" \"current f(x)\" \"best-so-far f(x)\" \"current af(x)+b\" \"best af(x)+b\"')
            for extra_info in self.extra_info_getters:
                f.write(f' {extra_info}')
            f.write('\n')

    def log(self, cur_evaluation, cur_fitness, best_so_far, loss_f, best_loss_f):
        with open(self.log_file_full_path, 'a') as f:
            f.write(f'{cur_evaluation} {loss_f} {best_loss_f} {cur_fitness} {best_so_far}')
            for fu in self.extra_info_getters:
                try:
                    extra_info = getattr(self.algorithm, fu)
                except Exception as e:
                    extra_info = 'None'
                f.write(f' {extra_info}')
            f.write('\n')
            self.last_line += 1

    def finish_logging(self):
        time_taken = time.process_time() - self.create_time
        with open(self.log_info_path, 'a') as f:
            f.write('%\n')
            f.write(f'{self.log_file_path}, {self.first_line}:{self.last_line}|{time_taken}\n')


class MyIOHFormatOnEveryEvaluationLogger2(ioh.iohcpp.logger.AbstractLogger):
    r"""
    This is a handler to perform the same kind of logging performed by Maria Laura in order to follow the
    IOH Analyzer guidelines.
    """
    def __init__(self, 
                 triggers = ..., 
                 properties = ..., 
                 root = ..., 
                 folder_name = ..., 
                 algorithm_name = ..., 
                 algorithm_info = ..., 
                 suite='unknown suite',
                 ):#store_positions = False):
        r"""
        Just the same constructor as the superclass (`ioh.logger.Analyzer`).
        """
        super().__init__(triggers, properties)
        
        # Fill up the properties
        self.root = root 
        self.folder_name = self.__generate_dir_name(folder_name)
        self.algorithm_name =algorithm_name
        self.algorithm_info = algorithm_info
        self.suite = suite

        self._setup_generated:bool = False
        self.create_time = time.process_time() 
    
    def __call__(self, logInfo:ioh.iohcpp.LogInfo, *args, **kwds):
        #print(logInfo,args,kwds)
        #return super().__call__(*args, **kwds)
        if not self._setup_generated:
            self._set_up_logger(fid= self.problem.problem_id,
                                iid =self.problem.instance,
                                dim = self.problem.n_variables,
                                func_name=self.problem.name)

            self._setup_generated = True
        
        # log the info
        self.log(logInfo.evaluations,
                 logInfo.y,
                 logInfo.y_best,
                 logInfo.y - logInfo.objective.y,
                 logInfo.y_best - logInfo.objective.y)
        
    
    def __generate_dir_name(self,name, x=0):
        while True:
            dir_name = (name + ('-' + str(x))).strip()
            if not os.path.exists(os.path.join(self.root,dir_name)):
                os.mkdir(dir_name)
                return dir_name
            else:
                x = x + 1
    
    def _set_up_logger(self, fid, iid, dim, func_name):
        self.log_info_path = f'{self.folder_name}/IOHprofiler_f{fid}_{func_name}.info'
        with open(self.log_info_path, 'a') as f:
            f.write(f'suite = \"{self.suite}\", funcId = {fid}, funcName = \"{func_name}\", DIM = {dim}, maximization = \"F\", algId = \"{self.algorithm_name}\", algInfo = \"{self.algorithm_info}\"\n')
        self.log_file_path = f'data_f{fid}_{func_name}/IOHprofiler_f{fid}_DIM{dim}.dat'
        self.log_file_full_path = f'{self.folder_name}/{self.log_file_path}'
        os.makedirs(os.path.dirname(self.log_file_full_path), exist_ok=True)
        self.first_line = 0
        self.last_line = 0
        with open(self.log_file_full_path, 'a') as f:
            f.write('\"function evaluation\" \"current f(x)\" \"best-so-far f(x)\" \"current af(x)+b\" \"best af(x)+b\"')
            for extra_info in self.extra_info_getters:
                f.write(f' {extra_info}')
            f.write('\n')
    
    def log(self, cur_evaluation, cur_fitness, best_so_far, loss_f, best_loss_f):
        with open(self.log_file_full_path, 'a') as f:
            f.write(f'{cur_evaluation} {loss_f} {best_loss_f} {cur_fitness} {best_so_far}')
            for fu in self.extra_info_getters:
                try:
                    extra_info = getattr(self.algorithm, fu)
                except Exception as e:
                    extra_info = 'None'
                f.write(f' {extra_info}')
            f.write('\n')
            self.last_line += 1
    
    @overload
    def watch(self, algorithm, extra_data)->None: ...

    def watch(self,obj:object,extra_props:list)->None:
        self.algorithm = obj
        self.extra_info_getters = extra_props
    
    def finish_logging(self):
        time_taken = time.process_time() - self.create_time
        with open(self.log_info_path, 'a') as f:
            f.write('%\n')
            f.write(f'{self.log_file_path}, {self.first_line}:{self.last_line}|{time_taken}\n')





        



