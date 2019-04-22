import argparse
import os
from datetime import datetime

import torch
from tensorboardX import SummaryWriter

from eco.config.config import config
from .model import ECOSeg
from .train.build import make_optimizer, make_lr_scheduler
from .train.comm import synchronize, get_rank
from .train.checkpoint import DetectronCheckpointer
from .data.build import make_data_loader
from .train.logger import setup_logger
from .train.trainer import do_train
from .train.inference import inference
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def train(local_rank, distributed):

    model = ECOSeg()
    device = torch.device(config.device)
    model.to(device)

    optimizer = make_optimizer(config, model)
    scheduler = make_lr_scheduler(config, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = config.output_dir

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        config, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(config.model_weight)
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        config,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = config.checkpoint_save_period

    TIMESTAMP = "{0:%Y-%m-%dT-%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard', TIMESTAMP))

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        writer,
    )

    writer.close()

    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox", "segm")    # bbox and mask

    output_folders = [None] * len(config.datasets_test)
    dataset_names = config.datasets_test
    if config.output_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(config.output_dir, "inference", dataset_name)
            os.makedirs(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(config, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch ECO Segmentation Training")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # Added by cxq 2019.4.1
    # Debug mode
    args.opts.extend(['SOLVER.IMS_PER_BATCH',2])
    args.opts.extend(['SOLVER.BASE_LR',0.0025])
    args.opts.extend(['SOLVER.MAX_ITER',720000])
    args.opts.extend(['SOLVER.STEPS',"(480000, 640000)"])
    args.opts.extend(['TEST.IMS_PER_BATCH',1])

    output_dir = config.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("ECOSeg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    model = train(args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(model, args.distributed)


if __name__ == "__main__":
    main()

