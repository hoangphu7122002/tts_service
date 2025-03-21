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
        self.wavs_raw_dir_path = self.data_dir_path/'wavs_raw'
        self.wavs_train_dir_path = self.data_dir_path/'wavs_train'
        self.alignment_file_path = self.data_dir_path/'alignment.txt'
        self.wer_reviewed_file_path = self.data_dir_path/'wer_reviewed.txt'
        self.metadata_file_path = self.data_dir_path/'metadata.txt'
        self.files_lists_dir_path = self.data_dir_path/'files_lists'
        self.train_list_file_path = self.files_lists_dir_path/'metadata_train.txt'
        self.val_list_file_path = self.files_lists_dir_path/'metadata_val.txt'

        # Experiments Paths
        self.experiments_dir_path = self.current_dir_path/'experiments'
        self.voice_name = os.path.basename(data_dir_path)
        self.outputs_dir_path = self.experiments_dir_path/self.voice_name
        self.tacotron_models_dir_path = self.outputs_dir_path/'tacotron_models'/version
        self.current_hparams_file_path = self.current_dir_path/'hparams.py'
        self.save_hparams_file_path = self.tacotron_models_dir_path/'hparams.py'
        self.save_train_list_file_path = self.tacotron_models_dir_path/'metadata_train.txt'
        self.save_val_list_file_path = self.tacotron_models_dir_path / 'metadata_val.txt'
        self.logs_dir_path = self.outputs_dir_path/'logs'/version
        self.samples_dir_path = self.outputs_dir_path/'tts_samples'


    def create_training_paths(self):
        os.makedirs(self.experiments_dir_path, exist_ok=True)
        os.chmod(self.experiments_dir_path, 0o775)
        os.makedirs(self.outputs_dir_path, exist_ok=True)
        os.chmod(self.outputs_dir_path, 0o775)
        os.makedirs(self.tacotron_models_dir_path, exist_ok=True)
        os.chmod(self.tacotron_models_dir_path, 0o775)
        os.makedirs(self.logs_dir_path, exist_ok=True)
        os.chmod(self.logs_dir_path, 0o775)

        copyfile(self.current_hparams_file_path, self.save_hparams_file_path)
        copyfile(self.train_list_file_path, self.save_train_list_file_path)
        copyfile(self.val_list_file_path, self.save_val_list_file_path)
