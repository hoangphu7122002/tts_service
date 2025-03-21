from pathlib import Path
import os
import sys
from datetime import date
from shutil import copyfile

class Paths:
    def __init__(self, data_dir_path, version):
        if not os.path.isdir(data_dir_path):
            sys.exit(f'No such directory: {data_dir_path}\nCheck data path')
        if not version:
            version = 'v.' + str(date.today().day) + str(date.today().month)
        self.current_dir_path = Path(os.path.dirname(os.path.realpath(__file__))).expanduser().resolve()

        # Data Paths
        self.data_dir_path = Path(data_dir_path).expanduser().resolve()
        self.wavs_dir_path = self.data_dir_path/'wavs'
        self.list_file_path = self.data_dir_path/'all.txt'
        self.train_list_file_path = self.data_dir_path/'train.txt'
        self.test_list_file_path = self.data_dir_path/'test.txt'

        # Experiments Paths
        self.experiments_dir_path = self.current_dir_path/'experiments'
        self.voice_name = os.path.basename(data_dir_path)
        self.outputs_dir_path = self.experiments_dir_path/self.voice_name       
        self.checkpoints_dir_path = self.outputs_dir_path/'checkpoints'/version
        self.configs_dir_path = self.outputs_dir_path/'configs'/version
        self.config_file_path = 'config.json'
        self.save_config_file_path = self.configs_dir_path/'config.json'
        self.save_train_list_file_path = self.configs_dir_path / 'train.txt'
        self.logs_dir_path = self.outputs_dir_path/'logs'/version
        
        # Samples Paths
        self.samples_dir_path = self.current_dir_path/'samples'
        

    def create_training_paths(self):
        os.makedirs(self.experiments_dir_path, exist_ok=True)
        os.chmod(self.experiments_dir_path, 0o775)
        os.makedirs(self.outputs_dir_path, exist_ok=True)
        os.chmod(self.outputs_dir_path, 0o775)
        os.makedirs(self.checkpoints_dir_path, exist_ok=True)
        os.chmod(self.checkpoints_dir_path, 0o775)
        os.makedirs(self.configs_dir_path, exist_ok=True)
        os.chmod(self.configs_dir_path, 0o775)        
        os.makedirs(self.logs_dir_path, exist_ok=True)
        os.chmod(self.logs_dir_path, 0o775)

        copyfile(self.config_file_path, self.save_config_file_path)
        copyfile(self.train_list_file_path, self.save_train_list_file_path)
 