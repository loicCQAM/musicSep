#!/usr/bin/env/python3
"""Recipe for training a neural speech separation system on wsjmix the
dataset. The system employs an encoder, a decoder, and a masking network.

To run this recipe, do the following:
> python train.py hparams/sepformer.yaml
> python train.py hparams/dualpath_rnn.yaml
> python train.py hparams/convtasnet.yaml

The experiment file is flexible enough to support different neural
networks. By properly changing the parameter files, you can try
different architectures. The script supports both wsj2mix and
wsj3mix.


Authors
 * Cem Subakan 2020
 * Mirco Ravanelli 2020
 * Samuele Cornell 2020
 * Mirko Bronzi 2020
 * Jianyuan Zhong 2020
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
import csv
import logging

from torch.utils.data import Dataset, DataLoader

# sys.path.append(os.path.expanduser("~") + "/demucs/demucs")
from raw import Rawset

from augment import FlipChannels, FlipSign, Remix, Shift

from utils_cem import center_trim
from tasnet import ConvTasNet
from torch.utils.data import Dataset, DataLoader

import museval
import musdb

from utils import is_empty_source


# Define training procedure
class Separation(sb.Brain):
    def compute_forward(self, targets, stage, inputs=None):
        """
        :param mixture: raw audio - dimension [batch_size, time]
        :param stage:
        :param init_params:
        :return:
        """

        if stage == sb.Stage.TRAIN:
            # predictions = self.compute_forward(inputs)
            # sources = center_trim(targets, estimates)
            targets = self.augment(targets)
            inputs = targets.sum(dim=1)

        est_source = self.hparams.convtasnet(inputs)
        print(est_source.shape)
        est_source2 = est_source / est_source.abs().max(dim=-1, keepdim=True)[0]
        print(est_source2.shape)

        # T changed after conv1d in encoder, fix it here
        T_origin = inputs.size(-1)
        T_est = est_source.size(-1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, T_origin - T_est))
        else:
            est_source = est_source[:, :, :, :T_origin]
        # [B, T, Number of speaker=2]
        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(predictions=predictions, targets=targets)

    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch[:, 0, :, :].to(self.device)
        targets = batch[:, 1:, :, :].to(self.device)

        if self.hparams.auto_mix_prec:
            with autocast():
                predictions = self.compute_forward(mixture)
                loss = self.compute_objectives(predictions, targets)

                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                else:
                    loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions, targets = self.compute_forward(targets, sb.Stage.TRAIN)

            predictions, targets = (
                predictions.permute(3, 0, 2, 1),
                targets.permute(3, 0, 2, 1),
            )
            predictions = predictions.reshape(
                predictions.size(0), -1, predictions.size(-1)
            )
            targets = targets.reshape(targets.size(0), -1, targets.size(-1))
            loss = self.compute_objectives(predictions, targets)

            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()

            if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm
                    )
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()

        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""

        if stage == sb.Stage.VALID:
            mixture = batch[:, 0, :, :].to(self.device)
            targets = batch[:, 1:, :, :].to(self.device)

            predictions, targets = self.compute_forward(targets, sb.Stage.TRAIN)

            predictions, targets = (
                predictions.permute(3, 0, 2, 1),
                targets.permute(3, 0, 2, 1),
            )
            predictions = predictions.reshape(
                predictions.size(0), -1, predictions.size(-1)
            )
            targets = targets.reshape(targets.size(0), -1, targets.size(-1))

            loss = self.compute_objectives(predictions, targets).mean()
        elif stage == sb.Stage.TEST:
            mixture = batch[0].to(self.device)
            targets = batch[1].to(self.device)
            lim = None
            from mir_eval.separation import bss_eval_sources

            with torch.no_grad():

                st = mixture.std()
                mean = mixture.mean()
                inp = mixture[:, :, :lim]  # - mean) / st

                inp = inp.to("cpu")

                predictions, _ = self.compute_forward(
                    targets=None, inputs=inp, stage=sb.Stage.TEST
                )

                # preds_max = predictions.max(-1, keepdim=True)[0]
                # predictions = predictions / preds_max
                # predictions = predictions * st  + mean

                mixture = mixture.to("cpu")
                predictions = predictions.to("cpu")
                targets = targets.to("cpu")

                i = self.testindex
                # track = self.test_mus.tracks[i]

                # scores = museval.evaluate(targets[:, :, :lim, :].squeeze(0), predictions.squeeze(0).permute(0, 2, 1))
                '''estimates = {
                    "vocals": predictions[0, -1, :, :].t().numpy(),
                    "drums": predictions[0, 0, :, :].t().numpy(),
                    "bass": predictions[0, 1, :, :].t().numpy(),
                    "accompaniment": predictions[0, 2, :, :].t().numpy(),
                }'''
                # scores = museval.eval_mus_track(track, estimates)

                has_zeros = False

                y1, y2 = targets[0, :, :lim, 0].numpy(), targets[0, :, :lim, 1].numpy()
                y_hat1, y_hat2 = predictions[0, :, 0, :].numpy(), predictions[0, :, 1, :].numpy()

                if is_empty_source(y1) or is_empty_source(y2) or is_empty_source(y_hat1) or is_empty_source(y_hat2):
                    has_zeros = True

                if not has_zeros:

                    scores1, _, _, _ = bss_eval_sources(
                        y1,
                        y_hat1
                    )
                    scores2, _, _, _ = bss_eval_sources(
                        y2,
                        y_hat2
                    )
                    scores = np.stack([scores1, scores2])

                    # scores = museval.evaluate(targets[0, :, :lim, :], predictions[0].permute(0, 2, 1))

                    self.all_scores.append(scores)

                    results_path = self.hparams.save_folder + "/audio_results"

                    if not os.path.exists(results_path):
                        os.makedirs(results_path)

                    if i < 5:
                        torchaudio.save(
                            filepath=results_path + "/mix_{}.wav".format(i),
                            src=mixture[0, :, :lim],
                            sample_rate=44100
                        )
                        torchaudio.save(
                            filepath=results_path + "/source1hat_{}.wav".format(i),
                            src=predictions[0, 0, :, :],
                            sample_rate=44100
                        )
                        torchaudio.save(
                            filepath=results_path + "/source2hat_{}.wav".format(i),
                            src=predictions[0, 1, :, :],
                            sample_rate=44100
                        )
                        torchaudio.save(
                            filepath=results_path + "/source3hat_{}.wav".format(i),
                            src=predictions[0, 2, :, :],
                            sample_rate=44100
                        )
                        torchaudio.save(
                            filepath=results_path + "/source4hat_{}.wav".format(i),
                            src=predictions[0, 3, :, :],
                            sample_rate=44100
                        )

                        torchaudio.save(
                            filepath=results_path + "/source1_{}.wav".format(i),
                            src=targets[0, 0, :lim, :].t(),
                            sample_rate=44100
                        )
                        torchaudio.save(
                            filepath=results_path + "/source2_{}.wav".format(i),
                            src=targets[0, 1, :lim, :].t(),
                            sample_rate=44100
                        )
                        torchaudio.save(
                            filepath=results_path + "/source3_{}.wav".format(i),
                            src=targets[0, 2, :lim, :].t(),
                            sample_rate=44100
                        )
                        torchaudio.save(
                            filepath=results_path + "/source4_{}.wav".format(i),
                            src=targets[0, 3, :lim, :].t(),
                            sample_rate=44100
                        )
                        self.testindex = self.testindex + 1

                loss = torch.tensor([0])

        return loss.detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"si-snr": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:

            # Learning rate annealing
            if isinstance(
                self.hparams.lr_scheduler, schedulers.ReduceLROnPlateau
            ):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss
                )
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]["lr"]

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"si-snr": stage_stats["si-snr"]}, min_keys=["si-snr"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(
                        self.hparams.min_shift, self.hparams.max_shift, (1,)
                    )
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0],), dims=1
                    )

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        # this applies the same speed perturb to each source
        if self.hparams.use_speedperturb_sameforeachsource:

            targets = targets.permute(0, 2, 1)
            targets = targets.reshape(-1, targets.shape[-1])
            wav_lens = torch.tensor([targets.shape[-1]] * targets.shape[0]).to(
                self.device
            )

            targets = self.hparams.speedperturb(targets, wav_lens)
            targets = targets.reshape(
                -1, self.hparams.num_spks, targets.shape[-1]
            )
            targets = targets.permute(0, 2, 1)

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length withing the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1,),
        ).item()
        targets = targets[
            :, randstart : randstart + self.hparams.training_signal_len, :
        ]
        mixture = mixture[
            :, randstart : randstart + self.hparams.training_signal_len
        ]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_loader):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder, "test_results.csv")

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ["snt_id", "sdr", "sdr_i", "si-snr", "si-snr_i"]

        with open(save_file, "w") as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture = batch[0]
                    snt_id = batch[0][0]
                    targets = batch[1:]
                    predictions, targets = self.compute_forward(
                        targets, sb.Stage.TEST
                    )

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [batch[0][1]] * self.hparams.num_spks, dim=-1
                    )
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets
                    )
                    sisnr_i = sisnr - sisnr_baseline

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        "snt_id": snt_id[0],
                        "sdr": sdr.mean(),
                        "sdr_i": sdr_i,
                        "si-snr": -sisnr.item(),
                        "si-snr_i": -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    "snt_id": "avg",
                    "sdr": np.array(all_sdrs).mean(),
                    "sdr_i": np.array(all_sdrs_i).mean(),
                    "si-snr": np.array(all_sisnrs).mean(),
                    "si-snr_i": np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info("Mean SISNR is {}".format(np.array(all_sisnrs).mean()))
        logger.info("Mean SISNRi is {}".format(np.array(all_sisnrs_i).mean()))
        logger.info("Mean SDR is {}".format(np.array(all_sdrs).mean()))
        logger.info("Mean SDRi is {}".format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        "saves the test audio (mixture, targets, and estimated sources) on disk"

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, "audio_results")
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}hat.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max()
            save_file = os.path.join(
                save_path, "item{}_source{}.wav".format(snt_id, ns + 1)
            )
            torchaudio.save(
                save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
            )

        # Mixture
        signal = mixture[1][0, :]
        signal = signal / signal.abs().max()
        save_file = os.path.join(save_path, "item{}_mix.wav".format(snt_id))
        torchaudio.save(
            save_file, signal.unsqueeze(0).cpu(), self.hparams.sample_rate
        )


class musdb_test_dataset(Dataset):
    def __init__(self, mus):
        self.mus = mus

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, idx):
        track = self.mus.tracks[idx]
        track.chunk_duration = 5.0
        track.chunk_start = np.random.uniform(0, track.duration - track.chunk_duration)
        x = torch.from_numpy(track.audio.T).float()
        y = torch.from_numpy(track.stems[1:, :, :]).float()
        return x, y


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Logger info
    logger = logging.getLogger(__name__)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data loaders
    # train_set = hparams["train_loader"]()
    # valid_set = hparams["valid_loader"]()
    # test_set = hparams["test_loader"]()

    test_mus = musdb.DB(hparams["musdb_path"], subsets=["test"])
    test_set = musdb_test_dataset(test_mus)
    test_loader = DataLoader(
        test_set, batch_size=hparams["N_batch"], shuffle=False
    )

    train_set = Rawset(
        os.path.join(hparams["musdb_raw_path"], "train"),
        samples=hparams["sample_rate"] * hparams["sequence_length"],
        channels=2,
        streams=[0, 1, 2, 3, 4],
        stride=hparams["sample_rate"],
    )

    train_loader = DataLoader(
        train_set, batch_size=hparams["N_batch"], shuffle=True
    )

    valid_set = Rawset(
        os.path.join(hparams["musdb_raw_path"], "valid"),
        samples=hparams["sample_rate"] * hparams["sequence_length"],
        channels=2,
        streams=[0, 1, 2, 3, 4],
        stride=hparams["sample_rate"],
    )

    valid_loader = DataLoader(
        valid_set, batch_size=hparams["N_batch"], shuffle=False
    )

    # Brain class initialization
    separator = Separation(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    separator.augment = torch.nn.Sequential(
        FlipSign(), FlipChannels(), Shift(hparams["sample_rate"]), Remix(group_size=1)
    ).to(hparams["device"])

    # re-initialize the parameters
    for module in separator.modules.values():
        separator.reset_layer_recursively(module)

    if not hparams["test_only"]:
        pass
        # Training
        separator.fit(
            separator.hparams.epoch_counter, train_loader, valid_loader
        )

    # Eval
    separator.modules = separator.modules.to('cpu')
    separator.testindex = 0
    separator.all_scores = []
    separator.evaluate(test_loader, min_key="si-snr")
    separator.test_mus = test_mus

    # Save Results
    ## separator.save_results(test_loader)
