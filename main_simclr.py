import os
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utils.utilities import (
    create_folder,
    get_filename,
    create_logging,
)
from models.models import *
from utils.pytorch_utils import (
    move_data_to_device,
    count_parameters,
)
from data.data_generator import (
    AudioSetViewsDataset,
    MultipleBalancedTrainSampler,
    collate_fn,
)
from models.losses import info_nce_loss
from augmentations.augmentations import *


def train(args):
    """Train AudioSet tagging model.

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_root = args.data_root
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    num_views = args.num_views
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    classes_num = args.embedding_dim
    temperature = args.temperature
    device = (
        torch.device("cuda")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )
    filename = args.filename

    num_workers = 8

    # Paths
    train_indexes_hdf5_path = os.path.join(
        data_root, "hdf5s", "indexes", "{}.h5".format(data_type)
    )

    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "simclr,sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        model_type,
    )
    create_folder(checkpoints_dir)

    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "simclr,sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax
        ),
        "data_type={}".format(data_type),
        model_type,
    )

    create_logging(logs_dir, filemode="w")
    logging.info(args)

    if "cuda" in str(device):
        logging.info("Using GPU.")
        device = "cuda"
    else:
        logging.info("Using CPU. Set --cuda flag to use GPU.")
        device = "cpu"

    # Model
    Model = eval(model_type)
    model = Model(
        sample_rate=sample_rate,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        fmin=fmin,
        fmax=fmax,
        classes_num=classes_num,
    )

    params_num = count_parameters(model)
    logging.info("Parameters num: {}".format(params_num))

    if pretrained_checkpoint_path:
        logging.info("Load pretrained model from {}".format(pretrained_checkpoint_path))
        checkpoint = torch.load(pretrained_checkpoint_path)
        model.load_state_dict(checkpoint["model"])

    # Dataset will be used by DataLoader later. Dataset takes a meta as input
    # and return a waveform and a target.
    transforms = Compose(
        [
            RandomGain(),
            Noise(min_snr=0.1, max_snr=0.3),
            RandomBackgroundNoise(
                noise_root="/home/klig/datasets/arabic-natural-audio",
                sample_rate=sample_rate,
                segment_size=sample_rate * 10,
                bank_size=1024,
                snr_dbs_range=[5, 10],
            ),
            RandomApply(RandomRIR(), p=0.8),
            RandomApply(RandomEncoder(sample_rate=sample_rate), p=0.8),
        ]
    )

    dataset = AudioSetViewsDataset(transforms=transforms, num_views=num_views)

    # Train sampler
    train_sampler = MultipleBalancedTrainSampler(
        indexes_hdf5_path=train_indexes_hdf5_path,
        batch_size=batch_size,
        num_repeat=2,
    )

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=int(early_stop * 0.9), gamma=0.1
    )

    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(
            workspace,
            "checkpoints",
            filename,
            "simclr,sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}".format(
                sample_rate, window_size, hop_size, mel_bins, fmin, fmax
            ),
            "data_type={}".format(data_type),
            model_type,
            "{}_iterations.pth".format(resume_iteration),
        )

        logging.info("Loading checkpoint {}".format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        train_sampler.load_state_dict(checkpoint["sampler"])
        iteration = checkpoint["iteration"]

    else:
        iteration = 0

    # Parallel
    logging.info("GPU number: {}".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if "cuda" in str(device):
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    time1 = time.time()

    for batch_data_dict in train_loader:
        """batch_data_dict: {
        'audio_name': (batch_size [*2 if mixup],),
        'waveform': (batch_size [*2 if mixup], clip_samples),
        'target': (batch_size [*2 if mixup], classes_num),
        (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """

        # Save model
        if iteration % 10000 == 0:
            checkpoint = {
                "iteration": iteration,
                "model": model.module.state_dict(),
                "sampler": train_sampler.state_dict(),
            }

            checkpoint_path = os.path.join(
                checkpoints_dir, "{}_iterations.pth".format(iteration)
            )

            torch.save(checkpoint, checkpoint_path)
            logging.info("Model saved to {}".format(checkpoint_path))

        # Forward
        model.train()

        waveforms = batch_data_dict["waveform"]
        waveforms = np.split(waveforms, waveforms.shape[1], axis=1)
        waveforms = np.concatenate(waveforms, axis=0).squeeze()
        waveforms = move_data_to_device(waveforms, device)

        features = model(waveforms)["clipwise_output"]

        # Loss
        logits, labels = info_nce_loss(
            features, n_views=num_views, temperature=temperature
        )
        loss = criterion(logits, labels)

        # Backward
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if iteration % 100 == 0:
            logging.info(
                "--- Iteration: {}, train time: {:.3f}, Loss {:.5f} ---".format(
                    iteration, time.time() - time1, loss.item()
                )
            )
            time1 = time.time()

        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of parser. ")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--data-root", type=str, required=True)
    parser_train.add_argument(
        "--workspace", type=str, required=False, default="results/"
    )
    parser_train.add_argument(
        "--data_type",
        type=str,
        default="full_train",
        choices=["balanced_train", "full_train"],
    )
    parser_train.add_argument("--sample_rate", type=int, default=8000)
    parser_train.add_argument("--window_size", type=int, default=256)
    parser_train.add_argument("--hop_size", type=int, default=80)
    parser_train.add_argument("--mel_bins", type=int, default=64)
    parser_train.add_argument("--fmin", type=int, default=50)
    parser_train.add_argument("--fmax", type=int, default=4000)
    parser_train.add_argument("--embedding_dim", type=int, default=512)
    parser_train.add_argument("--model_type", type=str, default="Cnn14_SIMCLR_8k")
    parser_train.add_argument("--num_views", type=int, default=2)
    parser_train.add_argument("--pretrained_checkpoint_path", type=str, default="")
    parser_train.add_argument("--batch_size", type=int, default=32)
    parser_train.add_argument("--temperature", type=float, default=0.1)
    parser_train.add_argument("--learning_rate", type=float, default=1e-4)
    parser_train.add_argument("--resume_iteration", type=int, default=0)
    parser_train.add_argument("--early_stop", type=int, default=1000000)
    parser_train.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == "train":
        train(args)

    else:
        raise Exception("Error argument!")
