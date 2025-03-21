import tensorflow as tf
from paths import Paths

def create_hparams_and_paths(hparams_string=None, verbose=False):
        hparams = tf.contrib.training.HParams(
                data='data/thonghoaithanh-22k',  # required
                version='v2-tempo1.1-20191111',  # default=v.DDMM

                ################################
                # Embedding Config             #
                ################################
                p_phone_mix=0.0,  # probability of mixing phone with letter (=1 for phone only...)
                spell_oov=True,  # for transcirpt
                phone_vn_train='dicts/phone_vn_south',  # required for phone embedding
                phone_oov_train='',
                eos='',
                punctuation='~,.*',
                letters='aáàạãảăắằặẵẳâấầậẫẩbcdđeéèẹẽẻêếềệễểghiíìịĩỉklmnoóòọõỏôốồộỗổơớờợỡởpqrstuúùụũủưứừựữửvxyýỳỵỹỷfjzw',
                coda_nucleus_and_semivowel=['iz', 'pc', 'nz', 'tc', 'ngz', 'kc', 'uz', 'mz', 'aa', 'ee', 'ea', 'oa',
                                            'aw', 'ie', 'uo', 'a', 'wa', 'oo', 'e', 'i', 'o', 'u', 'ow', 'uw', 'w'],

                ################################
                # Training Parameters          #
                ################################
                warm_start=True,
                ignore_layers=['embedding.weight'],
                fp16_run=True,
                batch_size=64,  # 32 or 64 (if not oom)
                distributed_run=False,
                epochs=1000,
                iters_per_checkpoint=5000,
                iters_per_valid=100,
                dynamic_loss_scaling=True,
                checkpoint_path='tacotron2_pretrained.pt',

                ################################
                # Audio Preprocess Parameters  #
                ################################
                norm_volume=True,
                volume_ratio=0.7,
                denoise=False,
                noise_frame=6,
                vad=False,
                vad_aggressiveness=1,
                trim_silence=True,
                trim_top_db=40,

                filter_audios=True,  # data selection
                longest_wav_in_seconds=12,
                limit_total_dur_in_hours=None,

                ################################
                # Audio Feature Parameters     #
                ################################
                max_wav_value=32768.0,
                sampling_rate=22050,
                filter_length=1024,
                hop_length=256,  # for 22k: 256, for 16k: 200
                win_length=1024,  # for 22k: 1024, for 16k: 800
                n_mel_channels=80,
                mel_fmin=55,  # for male: 55, for female: 95
                mel_fmax=7650.0,

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
                n_frames_per_step=1,
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
                use_last_lr=True,
                init_lr=1e-3,
                lr_decay=0.5,
                iter_start_decay=40000,
                iters_per_decay_lr=18000,
                final_lr=1e-4,
                eps=1e-6,
                weight_decay=1e-6,

                grad_clip_thresh=1.0,
                mask_padding=True
        )

        if hparams_string:
                tf.logging.info('Parsing command line hparams: %s', hparams_string)
                hparams.parse(hparams_string)

        if verbose:
                tf.logging.info('Final parsed hparams: %s', hparams.values())

        paths = Paths(hparams.data, hparams.version)

        return hparams, paths
