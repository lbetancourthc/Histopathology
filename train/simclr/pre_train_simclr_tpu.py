import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
from resnet_simclr import ResNetSimCLR
from utils.data_aug.gaussian_blur import GaussianBlur
from utils.data_aug.view_generator import (
    ContrastiveLearningViewGenerator,
    get_simclr_pipeline_transform,
)
import torch_xla.utils.serialization as xser
from utils.utils import accuracy_func, save_checkpoint


def init_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", metavar="DIR", default="/tmp/cifar", help="path to dataset"
    )
    parser.add_argument("--batch-size", default=64, type=int, metavar="N")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--learning-rate", default=0.00001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--num_cores",
        default=8,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument("--log_steps", default=100, type=int, help="Log every n steps")
    parser.add_argument("--metrics_debug", default=False, type=bool)
    parser.add_argument("--resume-epochs", type=int)
    parser.add_argument("--save-drive", default=False, action="store_true")

    return parser


def info_nce_loss(features, device):

    labels = torch.cat(
        [torch.arange(FLAGS.batch_size) for i in range(2)], dim=0
    )  # modifique a 2
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)

    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    TEMPERATURE = 0.07

    logits = logits / TEMPERATURE
    return logits, labels


parser = init_argparse()
FLAGS = parser.parse_args()


def train_resnet():

    train_dataset = datasets.ImageFolder(
        root="data/{}".format(FLAGS.data_dir),
        transform=ContrastiveLearningViewGenerator(
            get_simclr_pipeline_transform(224), n_views=2
        ),
    )

    train_sampler = None
    if xm.xrt_world_size() > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    torch.manual_seed(42)

    device = xm.xla_device()
    model = ResNetSimCLR(base_model="resnet50").to(device)

    if FLAGS.resume_epochs:
        print("Resuming Training ...")
        state_dict = xser.load(
            "models/net-DR-SimCLR-epoch-{}.pt".format(FLAGS.resume_epochs)
        )
        log = model.load_state_dict(state_dict, strict=False)
        print(log)

    # Scale learning rate to num cores
    learning_rate = FLAGS.learning_rate * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train_loop_fn(loader, epoch):
        tracker = xm.RateTracker()
        model.train()

        for x, (data, _) in enumerate(loader):
            optimizer.zero_grad()
            data = torch.cat(data, dim=0)
            output = model(data)
            logits, labels = info_nce_loss(output, device)
            loss = criterion(logits, labels)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(FLAGS.batch_size)

            top1, top5 = accuracy_func(logits, labels, topk=(1, 5))

            if x % FLAGS.log_steps == 0:
                print(
                    "[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}".format(
                        xm.get_ordinal(),
                        x,
                        loss.item(),
                        tracker.rate(),
                        tracker.global_rate(),
                        time.asctime(),
                    ),
                    flush=True,
                )
                print(f"Top1 accuracy: {top1[0]}")

    # Train loop
    accuracy = 0.0
    data, pred, target = None, None, None

    start_epoch = 0
    if FLAGS.resume_epochs:
        start_epoch = FLAGS.resume_epochs

    end_epoch = FLAGS.num_epochs

    train_device_loader = pl.MpDeviceLoader(train_loader, device)

    for epoch in range(start_epoch, end_epoch):
        # para_loader = pl.ParallelLoader(train_loader, [device])
        # train_loop_fn(para_loader.per_device_loader(device))
        train_loop_fn(train_device_loader, epoch)
        xm.master_print("Finished training epoch {}".format(epoch))

        if epoch % 2 == 0:
            model_name = "net-DR-SimCLR-epoch-{}.pt".format(epoch)
            if FLAGS.save_drive:
                model_name = (
                    "/content/drive/MyDrive/Colab Notebooks/SimCLR/models/SimCLR-1-DR-pytorch/"
                    + model_name
                )

                xm.save(model.state_dict(), model_name)
            else:
                xm.save(model.state_dict(), "models/" + model_name)

        if FLAGS.metrics_debug:
            xm.master_print(met.metrics_report(), flush=True)

    return accuracy, data, pred, target


# Start training processes
def _mp_fn(rank, flags):
    global FLAGS
    FLAGS = flags
    torch.set_default_tensor_type("torch.FloatTensor")
    accuracy, data, pred, target = train_resnet()


if __name__ == "__main__":
    parser = init_argparse()
    FLAGS = parser.parse_args()
    print(FLAGS)
    xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=FLAGS.num_cores, start_method="fork")