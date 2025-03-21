import tensorflow as tf
import os

hparams = tf.contrib.training.HParams(
        data='data/camhieu',
        version='v2',
        p_phone_mix=0.0,
        phone_vn='trained/dicts/phone_vn_south',
        phone_oov_train='',
        # syllable_oov='dicts/syllable_oov'
        eos='',
        punctuation=',.*',
        special='-',
        letters='aáàạãảăắằặẵẳâấầậẫẩbcdđeéèẹẽẻêếềệễểghiíìịĩỉklmnoóòọõỏôốồộỗổơớờợỡởpqrstuúùụũủưứừựữửvxyýỳỵỹỷfjzw',
        coda_nucleus_and_semivowel=['iz',  'pc',  'nz',  'tc',  'ngz',  'kc',  'uz',  'mz',  'aa',  'ee',  'ea',  'oa', 'aw',
                                    'ie',  'uo',  'a',  'wa',  'oo',  'e',  'i',  'o',  'u',  'ow',  'uw',  'w'],

        ################################
        # Training Parameters        #
        ################################
        warm_start=True,
        ignore_layers=['embedding.weight'],
        fp16_run=True,
        distributed_run=False,
        epochs=2000,
        iters_per_checkpoint=5000,
        iters_per_valid=100,
        lr_decay=0.5,
        epochs_per_decay_lr=200,
        dynamic_loss_scaling=True,
        use_last_lr=True,
        checkpoint_path='tacotron_pretrained',
        #checkpoint_path = 'last',
        ################################
        # Data Parameters             #
        ################################
        denoise=True,
        noise_frame=4,
        vad=True,
        vad_aggressiveness=2,
        trim_silence=True,
        trim_top_db=40,
        time_stretch=False,
        tempo=1.1,

        filter=True,
        remove_oov=False,
        longest_wav_in_seconds=12,
        limit_total_dur_in_hours=None,

        ################################
        # Feature Parameters             #
        ################################
        load_mel_from_disk=False,
        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################

        # Embedding parameters
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=2000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3,
        weight_decay=5e-5,
        grad_clip_thresh=1.0,
        batch_size=64,
        mask_padding=True  # set model's padded outputs to padded values
)


current_dir_path = os.path.dirname(os.path.realpath(__file__))

data_dir_path = hparams.data
wavs_raw_dir_path = os.path.join(data_dir_path, 'wavs_raw')
wavs_train_dir_path = os.path.join(data_dir_path, 'wavs_train')
metadata_file_path = os.path.join(data_dir_path, 'metadata.txt')
files_lists_dir_path = os.path.join(data_dir_path, 'files_lists')
train_list_file_path = os.path.join(files_lists_dir_path, 'train.txt')
val_list_file_path = os.path.join(files_lists_dir_path, 'val.txt')

experiments_dir_path = 'experiments'
speaker_name = os.path.basename(hparams.data)
outputs_dir_path = os.path.join(experiments_dir_path, speaker_name)
tacotron_models_dir_path = os.path.join(outputs_dir_path, 'tacotron_models', hparams.version)
current_hparams_file_path = os.path.join(current_dir_path, 'hparams.py')
save_hparams_file_path = os.path.join(tacotron_models_dir_path, 'hparams_' + hparams.version + '.py')
save_train_list_file_path = os.path.join(tacotron_models_dir_path, 'train_list_' + hparams.version)
logs_dir = os.path.join(outputs_dir_path, 'logs', hparams.version)
samples_dir = os.path.join(outputs_dir_path, 'tts_samples')

alignment_file_path = os.path.join(data_dir_path, 'alignment.txt')
