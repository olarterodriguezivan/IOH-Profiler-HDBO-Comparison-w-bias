import sys
import os
import json
import datetime
import subprocess

# #SBATCH --clusters=serial
class ExperimentEnvironment:
    SLURM_SCRIPT_TEMPLATE = '''#!/bin/bash

#SBATCH --job-name=##folder##
#SBATCH --array=0-##jobs_count##
#SBATCH --partition=gpu-medium
#SBATCH --mem=4096MB
#SBATCH --time=48:00:00
#SBATCH --mail-user=olarterodriguezi@vuw.leidenuniv.nl
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=##logs_out##
#SBATCH --error=##logs_err##


#// LOAD PYTHON
module load Python/3.10.4-GCCcore-11.3.0

#// ACTIVATE MODULE
source $HOME/BO_torch/bo-torch-run/bin/activate

num=##from_number##
FILE_ID=$((${SLURM_ARRAY_TASK_ID}+$num))

# Define the directory containing the config files
CONFIG_DIR="##folder_name_configs##"

# Find the file where NumExp-% matches TASK_ID
CONFIG_FILE=$(find "$CONFIG_DIR" -type f -name "*.json" | grep "NumExp-${FILE_ID}\.json")

# Check if a config file was found
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No matching config file found for NumExp-${FILE_ID}!"
    exit 1
fi

echo $CONFIG_FILE

python3 ../run_experiment.py $CONFIG_FILE
'''

    def __init__(self):
        now = datetime.datetime.now()
        suffix = now.strftime('%d-%m-%Y_%Hh%Mm%Ss')
        folder_name = 'run_' + suffix
        os.makedirs(folder_name, exist_ok=False)
        print(f'Experiment root is: {folder_name}')
        self.experiment_root = os.path.abspath(folder_name)
        self.__max_array_size = 1000
        self.__number_of_slurm_scripts = 0

    def set_up_by_experiment_config_file(self, experiment_config_file_name):
        self.__generate_configs(experiment_config_file_name)
        self.__create_log_dir()
        self.__generate_slurm_script()

    def __create_log_dir(self):
        self.logs_folder = os.path.join(self.experiment_root, 'logs')
        os.mkdir(self.logs_folder)

    def __generate_slurm_script(self):
        self.__number_of_slurm_scripts = 0
        logs_out = os.path.join(self.logs_folder, '%A_%a.out')
        logs_err = os.path.join(self.logs_folder, '%A_%a.err')
        script = ExperimentEnvironment.SLURM_SCRIPT_TEMPLATE
        script = script\
                .replace('##folder##', self.result_folder_prefix)\
                .replace('##logs_out##', logs_out)\
                .replace('##logs_err##', logs_err)\
                .replace('##folder_name_configs##', 
                         os.path.join(self.experiment_root,"configs"))
        offset = 0
        for _ in range(self.generated_configs // self.__max_array_size):
            with open(os.path.join(self.experiment_root, f'slurm{self.__number_of_slurm_scripts}.sh'), 'w') as f:
                f.write(script\
                        .replace('##from_number##', str(offset))\
                        .replace('##jobs_count##', str(self.__max_array_size - 1)))
            offset += self.__max_array_size
            self.__number_of_slurm_scripts += 1
        r = self.generated_configs % self.__max_array_size
        if r > 0:
            with open(os.path.join(self.experiment_root, f'slurm{self.__number_of_slurm_scripts}.sh'), 'w') as f:
                f.write(script\
                        .replace('##from_number##', str(offset))\
                        .replace('##jobs_count##', str(r - 1)))
            offset += r
            self.__number_of_slurm_scripts += 1

    def __generate_configs(self, experiment_config_file_name):
        with open(experiment_config_file_name, 'r') as f:
            config = json.load(f)
        self.result_folder_prefix = config['folder']
        fids = config['fids']
        iids = config['iids']
        dims = config['dims']
        reps = config['reps']
        logger_mode = config["logger"]
        sample_zero = bool(config["sample_zero"])
        if 'extra' not in config.keys():
            config['extra'] = ''
        optimizers = config['optimizers']
        lb, ub = config['lb'], config['ub']
        runs_number = len(optimizers) * len(fids) * len(iids) * len(dims) * reps
        cur_config_number = 0
        configs_dir = os.path.join(self.experiment_root, 'configs')
        os.makedirs(configs_dir, exist_ok=False)
        with open(os.path.join(self.experiment_root, 'description.json'), 'w') as f:
            json.dump(config, f, indent=4)
        for my_optimizer_name in optimizers:
            for fid in fids:
                for iid in iids:
                    for dim in dims:
                        # print(f'Ids for opt={my_optimizer_name}, fid={fid}, iid={iid}, dim={dim} are [{cur_config_number}, {cur_config_number+reps-1}]')
                        for rep in range(reps):
                            experiment_config = {
                                    'folder': f'{self.result_folder_prefix}_Opt-{my_optimizer_name}_F-{fid}_Id-{iid}_Dim-{dim}_Rep-{rep}_NumExp-{cur_config_number}',
                                    'opt': my_optimizer_name,
                                    'fid': fid,
                                    'iid': iid,
                                    'dim': dim,
                                    'seed': rep,
                                    'lb': lb,
                                    'ub': ub,
                                    'logger':logger_mode,
                                    'sample_zero':int(sample_zero)
                                    }
                            cur_config_file_name = f'Opt-{my_optimizer_name}_F-{fid}_Id-{iid}_Dim-{dim}_Rep-{rep}_NumExp-{cur_config_number}.json'
                            with open(os.path.join(configs_dir, cur_config_file_name), 'w') as f:
                                json.dump(experiment_config, f)
                            cur_config_number += 1
        print(f'Generated {cur_config_number} files')
        self.generated_configs = cur_config_number
    def is_slurm_available(self):
        try:
            subprocess.run(["sbatch", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            return False

    def print_helper(self):
        if self.is_slurm_available():
            print(f'cd {self.experiment_root} && for (( i=0; i<{self.__number_of_slurm_scripts}; ++i )); do sbatch slurm$i.sh; done')


def main(argv):
    env = ExperimentEnvironment()
    env.set_up_by_experiment_config_file(argv[1])
    env.print_helper()


if __name__ == '__main__':
    main(sys.argv)
