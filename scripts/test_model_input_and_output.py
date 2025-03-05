import argparse
import os
import pprint
import shutil
import _init_paths

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.config import get_model_name
from lib.core.function import train_integral
from lib.core.function import validate_integral, eval_integral
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger

import lib.core.integral_loss as loss
import lib.dataset as dataset
import lib.models as models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int,
                        default=8)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def main():
    best_perf = 0.0

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = models.pose3d_resnet.get_pose_net(config, is_train=True)
    # model = models.Vnet.VNet()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # model = model.to(device)

    # copy model file
    this_dir = os.path.dirname(__file__)

    args.cfg = r"D:\GitHub\WormID\EpipolarPose\experiments\wormnd\train-ss.yaml"
    shutil.copy2(
        args.cfg,
        final_output_dir
    )

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    loss_fn = eval('loss.'+config.LOSS.FN)
    criterion = loss_fn(num_joints=config.MODEL.NUM_JOINTS, norm=config.LOSS.NORM).cuda()

    # define training, validation and evaluation routines
    train = train_integral
    validate = validate_integral
    evaluate = eval_integral

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Resume from a trained model
    # if not(config.MODEL.RESUME is ''):
    if config.MODEL.RESUME != '':
        checkpoint = torch.load(config.MODEL.RESUME)
        if 'epoch' in checkpoint.keys():
            config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
        else:
            model.load_state_dict(checkpoint)
            logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))

    # Choose the dataset, either Human3.6M or mpii
    # ds = eval('dataset.'+config.DATASET.DATASET)
    ds = dataset.wormnd

    # Data loading code
    train_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TRAIN_SET,
        is_train=True
    )
    valid_dataset = ds(
        cfg=config,
        root=config.DATASET.ROOT,
        image_set=config.DATASET.TEST_SET,
        is_train=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    target_height = 256
    target_width = 480

    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        print(f'Epoch: {epoch}')

        # train for one epoch
        for i, data in enumerate(train_loader):
            # Debug print for data structure
            print(f"\nBatch {i} data structure:")
            print("Data type:", type(data))
            if isinstance(data, (list, tuple)):
                print("Data length:", len(data))
                print("Element types:", [type(item) for item in data])
                print("Tensor shapes:", [t.shape for t in data if isinstance(t, torch.Tensor)])
            
            # Convert input data to tensor if it's a list
            input = data
        
        
        # First do training step
        train(config, train_loader, model, criterion, optimizer, epoch)
        
        # Then update learning rate
        lr_scheduler.step()

        # evaluate on validation set
        for i, data in enumerate(valid_loader):
            input = data
             
            preds_in_patch_with_score = validate(valid_loader, model)

        perf_indicator = 0.0 
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

def test_model():

    best_perf = 0.0

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = models.pose3d_resnet.get_pose_net(config, is_train=True)
    # # model = models.Vnet.VNet()
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # print(device)
    # # model = model.to(device)
    #
    # # copy model file
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = VNet().to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    input = torch.randn(1, 1, 16, 256, 480)  # BCDHW
    input = torch.randn(1, 1, 16, 128, 128)  # BCDHW
    input = torch.randn(1, 2, 16, 128, 128)  # BCDHW
    input = input.to(device)
    out = model(input)
    print("output.shape:", out.shape)  # 4, 1, 8, 256, 256




if __name__ == '__main__':
    test_model()
# conda activate meta
# python scripts/train.py --cfg experiments/wormnd/train-ss.yaml

# C:/Users/jd/.conda/envs/meta/python.exe scripts/train.py --cfg experiments/wormnd/train-ss.yaml
# C:/Users/jd/.conda/envs/meta/python.exe scripts/test_model_input_and_output.py --cfg experiments/wormnd/train-ss.yaml