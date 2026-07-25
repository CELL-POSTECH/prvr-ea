import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Select the requested physical GPU before importing PyTorch or any module
# that may initialize CUDA.  This keeps --gpu effective for standalone runs.
for _arg_index, _arg_value in enumerate(sys.argv[:-1]):
    if _arg_value == '--gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_arg_index + 1]
        break
    if _arg_value.startswith('--gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = _arg_value.split('=', 1)[1]
        break

import argparse
import numpy as np
from tqdm import tqdm
import ipdb
import torch
import torch.nn as nn

from Configs.builder import get_configs
from Models.builder import get_models
from Datasets.builder import get_datasets
from Opts.builder import get_opts
from Losses.builder import get_losses
from Validations.builder import get_validations

from Utils.basic_utils import AverageMeter, BigFile, read_dict, log_config
from Utils.utils import set_seed, set_log, gpu, save_ckpt, load_ckpt
from collections import OrderedDict

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.dirname(os.path.dirname(root_path)))

parser = argparse.ArgumentParser(description="Partially Relevant Video Retrieval")
parser.add_argument('--model_name', default = 'N_np', type = str, help = 'specify gpu device')
parser.add_argument('--map_size', default = 128, type = int, help = 'specify gpu device')
parser.add_argument('--vl_coef', default = 0.1, type = float, help = 'specify gpu device')
parser.add_argument('--sim_thr', default = 0.5, type = float, help = 'specify gpu device')
parser.add_argument('--rkd_d_coef', default = 10, type = float, help = 'specify gpu device')
parser.add_argument('--rkd_a_coef', default = 20, type = float, help = 'specify gpu device')
parser.add_argument('--rkd_angle_chunk_size', default=None, type=int,
                    help='chunk RKD-angle over queries (0 keeps the original full tensor)')
parser.add_argument('-d', '--dataset_name', default='tvr', type=str, metavar='DATASET', help='dataset name', choices=['tvr', 'act', 'qvhighlight', 'cha', 'msrvtt', 'tvr_clip', 'act_clip', 'cha_clip', 'msrvtt_clip'])
parser.add_argument('--gpu', default = '0', type = str, help = 'specify gpu device')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--n_epoch', type=int, default=None, help='override configured epoch count')
parser.add_argument('--num_workers', type=int, default=None, help='override data loader worker count')
parser.add_argument('--multiGT', action='store_true', help='evaluate dense multi-positive ground truth')
args = parser.parse_args()
if args.multiGT:
    os.environ['PRVR_MULTI_GT'] = '1'
    os.environ['PRVR_MULTI_GT_DATASET'] = args.dataset_name

def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer, logger):

    if epoch >= cfg['hard_negative_start_epoch']:
        criterion.cfg['use_hard_negative'] = True
    else:
        criterion.cfg['use_hard_negative'] = False

    loss_meter = AverageMeter()
    loss_meters = OrderedDict(clip_nce=AverageMeter(), clip_trip=AverageMeter(),
                              frame_nce=AverageMeter(), frame_trip=AverageMeter(),
                              vidlvlctl=AverageMeter(), txt_rkd=AverageMeter(),
                              loss_overall=AverageMeter(), clip_num=AverageMeter())

    model.train()

    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                    unit="batch", dynamic_ncols=True)

    for idx, batch in enumerate(train_bar):
        batch = gpu(batch)
        optimizer.zero_grad()
        input_list = model(batch)
        loss, loss_dict = criterion(input_list, batch)
        
        loss.backward()
        optimizer.step()

        for k, v in loss_dict.items():
            loss_meters[k].update(float(v))
        loss_meter.update(loss.cpu().item())

        train_bar.set_description(
            'exp:{} e:{:2d} iter:{:3d} loss:{:.3f} clip:{:.1f}'.format(cfg['model_name'], epoch, idx, loss, loss_dict['clip_num']))
    loss_str = " ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()])
    logger.info('epoch: {:2d} {}'.format(epoch, loss_str))
    return loss_meter.avg


def val_one_epoch(epoch, context_dataloader, query_eval_loader, model, val_criterion, cfg, optimizer, best_val,
                  loss_meter, logger):
    val_meter, clip_val_meter, frame_val_meter = val_criterion(model, context_dataloader, query_eval_loader, logger)

    if val_meter[4] > best_val[4]:
        es = False
        sc = 'New Best Model !!!'
        best_val = val_meter
        save_ckpt(model, optimizer, cfg, os.path.join(cfg['model_root'], 'best.ckpt'), epoch, best_val)
    else:
        es = True
        sc = 'A Relative Failure Epoch'

    logger.info(
        '==========================================================================================================')
    logger.info('Epoch: {:2d}    {}'.format(epoch, sc))
    logger.info('Average Loss: {:.4f}'.format(loss_meter))
    logger.info(
        'R@1 5 10 100 Rsum: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(val_meter[0], val_meter[1], val_meter[2], val_meter[3], val_meter[4]))
    logger.info('Clip R@1 5 10 100 Rsum: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(clip_val_meter[0], clip_val_meter[1], clip_val_meter[2], clip_val_meter[3], clip_val_meter[4]))
    logger.info('Frame R@1 5 10 100 Rsum: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(frame_val_meter[0], frame_val_meter[1], frame_val_meter[2], frame_val_meter[3], frame_val_meter[4]))
    logger.info(
        'Best: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(best_val[0], best_val[1], best_val[2], best_val[3],
                                                              best_val[4]))
    logger.info(
        '==========================================================================================================')

    return val_meter, best_val, es


def validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, resume):
    val_meter, clip_val_meter, frame_val_meter = val_criterion(model, context_dataloader, query_eval_loader, logger)

    logger.info('==========================================================================================================')
    logger.info('Testing from: {}'.format(resume))
    logger.info('R@1 5 10 100 Rsum: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(val_meter[0], val_meter[1], val_meter[2], val_meter[3], val_meter[4]))
    logger.info(
        'Clip R@1 5 10 100 Rsum: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(clip_val_meter[0], clip_val_meter[1],
                                                                                clip_val_meter[2], clip_val_meter[3],
                                                                                clip_val_meter[4]))
    logger.info(
        'Frame R@1 5 10 100 Rsum: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}'.format(frame_val_meter[0], frame_val_meter[1],
                                                                                 frame_val_meter[2], frame_val_meter[3],
                                                                                 frame_val_meter[4]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info(
        '==========================================================================================================')

import yaml
def main():
    cfg = get_configs(args.dataset_name)
    if args.n_epoch is not None:
        cfg['n_epoch'] = args.n_epoch
    if args.num_workers is not None:
        cfg['num_workers'] = args.num_workers
    cfg['model_name'] = args.model_name
    cfg['map_size'] = args.map_size
    cfg['vl_coef'] = args.vl_coef
    cfg['sim_thr'] = args.sim_thr
    cfg['rkd_d_coef'] = args.rkd_d_coef
    cfg['rkd_a_coef'] = args.rkd_a_coef
    # MSRVTT has exactly 20 captions per video. With the original batch size
    # of 128, RKD-angle therefore receives 2,560 queries and its dense N^3
    # tensor cannot fit in 24 GiB.  A 32-query block keeps the exact loss while
    # using the available memory safely. Other datasets retain the original
    # dense calculation unless an explicit override is supplied.
    if args.rkd_angle_chunk_size is None:
        cfg['rkd_angle_chunk_size'] = 32 if cfg['collection'] == 'msrvtt' else 0
    else:
        cfg['rkd_angle_chunk_size'] = args.rkd_angle_chunk_size
    # The feature-mode-specific project-local path is configured by the local
    # dataset adapter.  Keep that layout after applying MSC's CLI model name.
    cfg['model_root'] = os.path.join(
        cfg['root'], 'results', cfg['feature_mode'], cfg['dataset_name'],
        cfg['model_name'])
    cfg['ckpt_path'] = os.path.join(cfg['model_root'], 'ckpt')
    cfg['tb_dir'] = cfg['model_root']
    if not os.path.exists(cfg['model_root']):
        os.makedirs(cfg['model_root'], exist_ok=True)
    if not os.path.exists(cfg['ckpt_path']):
        os.makedirs(cfg['ckpt_path'], exist_ok=True)
    with open(os.path.join(cfg['model_root'], 'hyperparams.yaml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file)
    # set logging
    logger = set_log(cfg['model_root'], 'log.txt')
    logger.info('Partially Relevant Video Retrieval Training: {}'.format(cfg['dataset_name']))

    # Select the GPU before set_seed(), which calls CUDA seed APIs.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = range(torch.cuda.device_count())
    logger.info('used gpu: {}'.format(args.gpu))

    # set seed
    set_seed(cfg['seed'])
    logger.info('set seed: {}'.format(cfg['seed']))

    # hyper parameter

    logger.info('Hyper Parameter ......')
    logger.info(cfg)

    # dataset
    logger.info('Loading Data ......')
    cfg, train_loader, context_dataloader, query_eval_loader = get_datasets(cfg)

    # model
    logger.info('Loading Model ......') 
    model = get_models(cfg)

    # initial
    current_epoch = -1
    es_cnt = 0
    best_val = [0., 0., 0., 0., 0.]
    if args.resume != '':
        logger.info('Resume from {}'.format(args.resume))
        _, model_state_dict, optimizer_state_dict, current_epoch, best_val = load_ckpt(args.resume)
        model.load_state_dict(model_state_dict)
    model = model.cuda()
    if len(device_ids) > 1:
        model = nn.DataParallel(model)
    
    criterion = get_losses(cfg)
    val_criterion = get_validations(cfg)

    if args.eval:
        if args.resume == '':
            logger.info('No trained ckpt load !!!') 
        else:
            with torch.no_grad():
                validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, args.resume)
        exit(0)

    optimizer = get_opts(cfg, model, train_loader)
    if args.resume != '':
        optimizer.load_state_dict(optimizer_state_dict)

    for epoch in range(current_epoch + 1, cfg['n_epoch']):

        ############## train
        loss_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer, logger)

        ############## val
        with torch.no_grad():
            val_meter, best_val, es = val_one_epoch(epoch, context_dataloader, query_eval_loader, model, 
                    val_criterion, cfg, optimizer, best_val, loss_meter, logger)

        ############## early stop
        if not es:
            es_cnt = 0
        else:
            es_cnt += 1
            if cfg['max_es_cnt'] != -1 and es_cnt > cfg['max_es_cnt']:  # early stop
                logger.info('Early Stop !!!') 
                exit(0)


if __name__ == '__main__':
    main()
