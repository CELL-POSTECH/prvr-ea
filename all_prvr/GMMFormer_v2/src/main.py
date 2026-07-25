import os
import argparse
import numpy as np
import random
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import time
from tqdm import tqdm
import ipdb
import pickle

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


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
sys.path.insert(0, os.path.dirname(os.path.dirname(root_path)))

parser = argparse.ArgumentParser(description="Partially Relevant Video Retrieval")
parser.add_argument(
    '-d', '--dataset_name', default='tvr', type=str, metavar='DATASET', help='dataset name', 
    choices=['tvr', 'act', 'cha', 'msrvtt', 'tvr_clip', 'act_clip', 'cha_clip', 'msrvtt_clip']
)
parser.add_argument(
    '--gpu', default = '0', type = str, help = 'specify gpu device'
    )
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--n_epoch', type=int, default=None, help='override configured epoch count')
parser.add_argument('--num_workers', type=int, default=None, help='override data loader worker count')
parser.add_argument('--eval_query_bsz', type=int, default=None, help='override evaluation query batch size')
parser.add_argument('--multiGT', action='store_true', help='evaluate dense multi-positive ground truth')
parser.add_argument('--ann_benchmark', action='store_true', help='run ANN retrieval benchmark (eval only)')
parser.add_argument('--ann_build_context_bank', action='store_true', help='encode test videos and save ANN context bank')
parser.add_argument('--ann_context_bank', default='', type=str, help='context-bank .pt path; defaults beside checkpoint')
parser.add_argument('--ann_index_dir', default='', type=str, help='directory for persistent FAISS indices; defaults beside checkpoint')
parser.add_argument('--ann_rebuild_index', action='store_true', help='ignore a matching persisted FAISS index and rebuild it')
parser.add_argument('--ann_index', default='ivf', choices=['origin', 'flat_full', 'ivf', 'ivf-gpu', 'hnsw'])
parser.add_argument('--ann_clip_raw_k', default=832, type=int)
parser.add_argument('--ann_frame_raw_k', default=2948, type=int)
parser.add_argument('--ann_candidate_k', default=30, type=int)
parser.add_argument('--ann_nlist', default=0, type=int,
                    help='IVF list count; 0=per-branch 2^floor(log2(sqrt(raw corpus)))')
parser.add_argument('--ann_nprobe', default=64, type=int)
parser.add_argument('--ann_ef_search', default=256, type=int)
parser.add_argument('--ann_output', default='', type=str)
parser.add_argument('--ann_max_queries', default=0, type=int, help='optional ANN smoke-test query limit (0 = all)')
parser.add_argument('--synthetic_speed_eval', action='store_true', help='run synthetic 512-D full-pipeline speed evaluation (eval only)')
parser.add_argument('--synthetic_data_root', default='', type=str, help='synthetic data root; defaults to PRVR_DATA_ROOT')
parser.add_argument('--synthetic_context_ids', default='', type=str, help='one synthetic video id per line')
parser.add_argument('--synthetic_context_shard_videos', default=45234, type=int)
parser.add_argument('--synthetic_chunk_vector_budget', default=7237555, type=int)
parser.add_argument('--synthetic_topk', default=10, type=int)
parser.add_argument('--synthetic_output', default='', type=str, help='CSV output path for synthetic speed row')
args = parser.parse_args()
if args.multiGT:
    os.environ['PRVR_MULTI_GT'] = '1'
    os.environ['PRVR_MULTI_GT_DATASET'] = args.dataset_name


def train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer):

    if epoch >= cfg['hard_negative_start_epoch']:
        criterion.cfg['use_hard_negative'] = True
    else:
        criterion.cfg['use_hard_negative'] = False

    loss_meter = AverageMeter()

    model.train()

    train_bar = tqdm(train_loader, desc="epoch " + str(epoch), total=len(train_loader),
                    unit="batch", dynamic_ncols=True)

    for idx, batch in enumerate(train_bar):

        batch = gpu(batch)

        optimizer.zero_grad()

        input_list = model(batch)

        loss = criterion(input_list, batch)
        
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.cpu().item())

        train_bar.set_description('exp: {} epoch:{:2d} iter:{:3d} loss:{:.4f}'.format(cfg['model_name'], epoch, idx, loss))

    return loss_meter.avg


def val_one_epoch(epoch, context_dataloader, query_eval_loader, model, val_criterion, cfg, optimizer, best_val, loss_meter, logger):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)

    if val_meter[4] > best_val[4]:
        es = False
        sc = 'New Best Model !!!'
        best_val = val_meter
        save_ckpt(model, optimizer, cfg, os.path.join(cfg['model_root'], 'best.ckpt'), epoch, best_val)
    else:
        es = True
        sc = 'A Relative Failure Epoch'
                
    logger.info('==========================================================================================================')
    logger.info('Epoch: {:2d}    {}'.format(epoch, sc))
    logger.info('Average Loss: {:.4f}'.format(loss_meter))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('Best: R@1: {:.1f} R@5: {:.1f} R@10: {:.1f} R@100: {:.1f} Rsum: {:.1f}'.format(best_val[0], best_val[1], best_val[2], best_val[3], best_val[4]))
    logger.info('==========================================================================================================')
        
    return val_meter, best_val, es


def validation(context_dataloader, query_eval_loader, model, val_criterion, cfg, logger, resume):

    val_meter = val_criterion(model, context_dataloader, query_eval_loader)
    
    logger.info('==========================================================================================================')
    logger.info('Testing from: {}'.format(resume))
    logger.info('R@1: {:.1f}'.format(val_meter[0]))
    logger.info('R@5: {:.1f}'.format(val_meter[1]))
    logger.info('R@10: {:.1f}'.format(val_meter[2]))
    logger.info('R@100: {:.1f}'.format(val_meter[3]))
    logger.info('Rsum: {:.1f}'.format(val_meter[4]))
    logger.info('==========================================================================================================')


def main():
    cfg = get_configs(args.dataset_name)
    if args.n_epoch is not None:
        cfg['n_epoch'] = args.n_epoch
    if args.num_workers is not None:
        cfg['num_workers'] = args.num_workers
    if args.eval_query_bsz is not None:
        cfg['eval_query_bsz'] = args.eval_query_bsz

    # set logging
    logger = set_log(cfg['model_root'], 'log.txt')
    logger.info('Partially Relevant Video Retrieval Training: {}'.format(cfg['dataset_name']))

    # set seed
    set_seed(cfg['seed'])
    logger.info('set seed: {}'.format(cfg['seed']))

    # hyper parameter
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device_ids = range(torch.cuda.device_count())
    logger.info('used gpu: {}'.format(args.gpu))

    logger.info('Hyper Parameter ......')
    logger.info(cfg)

    # dataset
    logger.info('Loading Data ......')
    if args.synthetic_speed_eval:
        if not args.eval:
            raise ValueError('--synthetic_speed_eval requires --eval')
        if not args.dataset_name.endswith('_clip'):
            raise ValueError('--synthetic_speed_eval requires a *_clip dataset mode')
        from Datasets.synthetic_speed import get_synthetic_speed_datasets
        test_context_dataloader, test_query_eval_loader = get_synthetic_speed_datasets(cfg, args)
        train_loader = context_dataloader = query_eval_loader = None
    else:
        cfg, train_loader, context_dataloader, query_eval_loader, test_context_dataloader, test_query_eval_loader = get_datasets(cfg)

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
                if args.synthetic_speed_eval:
                    from Validations.synthetic_speed import run_synthetic_speed_eval
                    run_synthetic_speed_eval(model, test_context_dataloader, test_query_eval_loader, cfg, args, logger)
                elif args.ann_build_context_bank or args.ann_benchmark:
                    from Validations.ann_benchmark import build_context_bank, run_ann_benchmark
                    if args.ann_build_context_bank:
                        bank_path = build_context_bank(
                            model, val_criterion, test_context_dataloader, cfg, args.resume,
                            args.ann_context_bank, logger)
                        logger.info('ANN context bank written: %s', bank_path)
                    if args.ann_benchmark:
                        run_ann_benchmark(
                            model, val_criterion, test_context_dataloader,
                            test_query_eval_loader, cfg, args, logger)
                else:
                    validation(test_context_dataloader, test_query_eval_loader, model, val_criterion, cfg, logger, args.resume)
        exit(0)

    optimizer = get_opts(cfg, model, train_loader)
    if args.resume != '':
        optimizer.load_state_dict(optimizer_state_dict)

    for epoch in range(current_epoch + 1, cfg['n_epoch']):

        ############## train
        loss_meter = train_one_epoch(epoch, train_loader, model, criterion, cfg, optimizer)

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
