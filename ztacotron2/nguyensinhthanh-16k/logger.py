import random
import torch
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np



def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, learning_rate,
                     iteration):

        self.add_scalar("learning_rate", learning_rate, iteration)
        self.add_scalar("training_loss", reduced_loss, iteration)



    def log_validation(self, reduced_loss, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)

        self.add_image(
            "alignment_random",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration)

        '''
        self.add_image(
            "mel_predicted_random",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        
        self.add_image(
            "mel_target_random",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration)
        '''
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[1].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[1]).data.cpu().numpy()),
            iteration)

        self.add_image(
            "gate_random",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)

        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[1].data.cpu().numpy().T),
            iteration)
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[1].data.cpu().numpy()), 0)
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[1].data.cpu().numpy()),
            iteration)