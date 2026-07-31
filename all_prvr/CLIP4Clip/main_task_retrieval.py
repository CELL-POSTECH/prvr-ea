from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
import json
from collections import OrderedDict
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT

torch.distributed.init_process_group(backend="nccl")

global logger

def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')
    parser.add_argument('--frame_root', type=str, default=None,
                        help='Root containing per-video raw-frame directories for raw_* datatypes.')
    parser.add_argument('--caption_root', type=str, default=None,
                        help='Root containing {dataset}{split}.caption.txt files for raw_* datatypes.')
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='Optional raw_* training-sample limit; 0 uses the complete split.')
    parser.add_argument('--max_eval_samples', type=int, default=0,
                        help='Optional raw_* evaluation-sample limit; 0 uses the complete split.')
    parser.add_argument('--multi_gt_eval', action='store_true',
                        help='Evaluate denseval.gt.jsonl with macro multi-positive recall.')
    parser.add_argument('--multi_gt_caption_file', default=None, type=str,
                        help='Optional denseval caption file override.')
    parser.add_argument('--multi_gt_file', default=None, type=str,
                        help='Optional denseval.gt.jsonl override.')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--chunk_size', type=int, default=0,
                        help='Raw-frame zero-shot eval only. If > 0, split the original ordered '
                             'raw-frame sequence before max_frames sampling, mean-pool each chunk, '
                             'and use the maximum chunk score as the parent-video score.')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def _chunked_meanp_similarity(model, sequence_output, visual_output, video_mask, chunk_size):
    """Return parent-video scores after fixed-size chunk mean pooling.

    This is deliberately the same math as CLIP4Clip's loose ``meanP`` path
    within each chunk: L2-normalize frame features, mean pool valid frames,
    L2-normalize the chunk feature, then take the CLIP dot product.  A video
    may yield several chunks, but their scores are max-reduced *before* the
    retrieval ranking so a parent video occupies one ranked slot.
    """
    if not model.loose_type or model.sim_header != "meanP":
        raise ValueError("--chunk_size requires --loose_type --sim_header meanP")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive in chunked similarity")

    # Match modules/modeling.py::_loose_similarity(meanP) exactly up to the
    # point where it would pool an entire video.
    # The raw-frame loader retains a singleton ``pair`` dimension in the
    # cached mask.  ``get_similarity_logits(..., shaped=False)`` flattens it
    # internally; do the same here before addressing the temporal axis.
    video_mask = video_mask.view(-1, video_mask.shape[-1])
    visual_output = visual_output.contiguous()
    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sequence_output = sequence_output.squeeze(1)
    sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    logit_scale = model.clip.logit_scale.exp()

    batch_v, frame_count, _ = visual_output.shape
    parent_scores = torch.full(
        (sequence_output.size(0), batch_v), -float("inf"),
        dtype=visual_output.dtype, device=visual_output.device,
    )
    for start in range(0, frame_count, chunk_size):
        end = min(start + chunk_size, frame_count)
        chunk_mask = video_mask[:, start:end].to(dtype=visual_output.dtype).unsqueeze(-1)
        valid = chunk_mask.sum(dim=1)
        valid_rows = valid.squeeze(-1) > 0
        if not torch.any(valid_rows):
            continue
        chunk_output = (visual_output[:, start:end] * chunk_mask).sum(dim=1)
        chunk_output = chunk_output / valid.clamp_min(1.0)
        chunk_output = chunk_output / chunk_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chunk_scores = logit_scale * torch.matmul(sequence_output, chunk_output.t())
        chunk_scores[:, ~valid_rows] = -float("inf")
        parent_scores = torch.maximum(parent_scores, chunk_scores)
    return parent_scores


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list,
                       batch_visual_output_list, chunk_size=0,
                       chunk_parent_indices=None, parent_video_num=None):
    """Compute retrieval scores, optionally max-reducing raw-frame chunks.

    ``chunk_parent_indices`` maps each cached visual chunk to its parent video.
    In that mode each visual item is already one raw-frame chunk, so it uses
    ordinary CLIP4Clip meanP scoring and max-reduces chunk scores per parent.
    """
    prechunked = chunk_parent_indices is not None
    if prechunked and (parent_video_num is None or len(chunk_parent_indices) != len(batch_list_v)):
        raise ValueError("pre-chunked similarity requires parent indices for every visual batch")

    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        if prechunked:
            each_parent_score = torch.full(
                (sequence_output.shape[0], int(parent_video_num)), -float("inf"),
                dtype=sequence_output.dtype, device=sequence_output.device,
            )
        else:
            each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            if chunk_size > 0 and not prechunked:
                b1b2_logits = _chunked_meanp_similarity(
                    model, sequence_output, visual_output, video_mask, chunk_size)
            else:
                b1b2_logits, *_tmp = model.get_similarity_logits(
                    sequence_output, visual_output, input_mask, video_mask,
                    loose_type=model.loose_type)
            if prechunked:
                parent_indices = chunk_parent_indices[idx2]
                scatter_indices = parent_indices.unsqueeze(0).expand(b1b2_logits.shape[0], -1)
                each_parent_score.scatter_reduce_(
                    1, scatter_indices, b1b2_logits, reduce="amax", include_self=True
                )
            else:
                each_row.append(b1b2_logits.cpu().detach().numpy())
        if prechunked:
            sim_matrix.append(each_parent_score.cpu().detach().numpy())
        else:
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)
    return sim_matrix


def _prechunked_parent_similarity(model, sequence_outputs, chunk_outputs,
                                  parent_indices, parent_video_num,
                                  query_batch_size):
    """Score all pre-encoded raw chunks, then max-reduce to parent videos.

    The previous implementation nested text batches and chunk batches.  For
    TVR chunk=10 this meant roughly 341 text batches x 1,300 chunk batches,
    i.e. hundreds of thousands of tiny GPU similarity calls.  ``meanP`` is
    just normalized dot-product scoring, so concatenate the chunk bank and
    process one query batch against it at a time instead.  The score and the
    parent-video max reduction are mathematically unchanged.
    """
    if not sequence_outputs or not chunk_outputs:
        raise ValueError("pre-chunked similarity requires non-empty text and chunk caches")

    sequence_output = torch.cat(sequence_outputs, dim=0).squeeze(1)
    sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    chunk_output = torch.cat(chunk_outputs, dim=0)
    chunk_output = chunk_output / chunk_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if chunk_output.size(0) != parent_indices.numel():
        raise ValueError("pre-chunk output and parent-index counts differ")

    logit_scale = model.clip.logit_scale.exp()
    total_queries = int(sequence_output.size(0))
    step = max(1, int(query_batch_size))
    report_step, next_report = max(1, total_queries // 20), max(1, total_queries // 20)
    started = time.time()
    sim_rows = []
    for start in range(0, total_queries, step):
        stop = min(start + step, total_queries)
        chunk_scores = logit_scale * torch.matmul(sequence_output[start:stop], chunk_output.t())
        parent_scores = torch.full(
            (stop - start, int(parent_video_num)), -float("inf"),
            dtype=chunk_scores.dtype, device=chunk_scores.device,
        )
        scatter_indices = parent_indices.unsqueeze(0).expand(stop - start, -1)
        parent_scores.scatter_reduce_(1, scatter_indices, chunk_scores, reduce="amax", include_self=True)
        sim_rows.append(parent_scores.cpu().numpy())

        if stop >= next_report or stop == total_queries:
            elapsed = time.time() - started
            rate = stop / max(elapsed, 1e-6)
            eta = (total_queries - stop) / max(rate, 1e-6)
            logger.info("Chunk retrieval: %d/%d queries (%.1f%%), elapsed %.1fs, ETA %.1fs",
                        stop, total_queries, 100.0 * stop / total_queries, elapsed, eta)
            while next_report <= stop:
                next_report += report_step
    return np.concatenate(tuple(sim_rows), axis=0)


def _load_multi_gt_mapping(gt_path):
    mapping = OrderedDict()
    with open(gt_path, "r", encoding="utf-8") as reader:
        for line_no, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            query_id = item.get("query_id")
            video_ids = item.get("gt_video_ids") or item.get("video_ids") or item.get("videos")
            if not query_id or video_ids is None:
                raise ValueError("invalid multi-GT line {} in {}".format(line_no, gt_path))
            mapping[query_id] = list(video_ids)
    return mapping


def _multi_gt_text_to_video_metrics(scores, query_ids, video_ids, gt_path):
    """Macro positive coverage: |top-K ∩ GT(q)| / |GT(q)|, averaged over q."""
    mapping = _load_multi_gt_mapping(gt_path)
    video_to_index = {video_id: index for index, video_id in enumerate(video_ids)}
    recalls = {1: [], 5: [], 10: []}
    evaluated = 0
    for row_index, query_id in enumerate(query_ids):
        if query_id not in mapping:
            raise ValueError("query {} is absent from {}".format(query_id, gt_path))
        positives = [video_to_index[v] for v in mapping[query_id] if v in video_to_index]
        missing = set(mapping[query_id]) - set(video_to_index)
        if missing:
            raise ValueError("multi-GT candidates missing from evaluation collection for {}: {}".format(
                query_id, sorted(missing)[:5]))
        if not positives:
            continue
        evaluated += 1
        ranking = np.argsort(-scores[row_index])
        positive_set = set(positives)
        for k in recalls:
            recalls[k].append(len(set(ranking[:k]) & positive_set) / float(len(positive_set)))
    if not evaluated:
        raise ValueError("no multi-GT queries had candidate positives: {}".format(gt_path))
    metrics = {"R{}".format(k): float(np.mean(values) * 100.0) for k, values in recalls.items()}
    metrics["evaluated"] = evaluated
    return metrics

def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))
    if args.chunk_size < 0:
        raise ValueError("--chunk_size must be 0 (original) or a positive integer")
    if args.chunk_size > 0:
        if not args.datatype.startswith("raw_"):
            raise ValueError("--chunk_size is supported only for raw-frame datatypes")
        if args.sim_header != "meanP" or not args.loose_type:
            raise ValueError("--chunk_size requires --loose_type --sim_header meanP")
        logger.info("Pre-sampling chunked raw-frame evaluation: chunk_size=%d; "
                    "parent-video score=max(chunk scores)", args.chunk_size)

    raw_prechunked = args.chunk_size > 0 and args.datatype.startswith("raw_")
    if raw_prechunked and n_gpu > 1:
        raise ValueError("raw-frame pre-chunked evaluation currently supports one GPU")

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        chunk_parent_indices = []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        text_started = time.time()
        text_report_step = max(1, len(test_dataloader) // 20)
        text_next_report = text_report_step
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask = batch

            if raw_prechunked:
                # The raw-frame dataset returns dummy visuals here. Chunk
                # visuals are loaded once per parent video below, before any
                # max_frames sampling, rather than once per caption.
                sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
            elif multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model.get_sequence_visual_output(input_ids, segment_ids, input_mask, video, video_mask)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            completed = bid + 1
            if completed >= text_next_report or completed == len(test_dataloader):
                elapsed = time.time() - text_started
                rate = completed / max(elapsed, 1e-6)
                eta = (len(test_dataloader) - completed) / max(rate, 1e-6)
                logger.info("Text encoding: %d/%d batches (%.1f%%), elapsed %.1fs, ETA %.1fs",
                            completed, len(test_dataloader),
                            100.0 * completed / len(test_dataloader), elapsed, eta)
                while text_next_report <= completed:
                    text_next_report += text_report_step

        if raw_prechunked:
            pending_videos, pending_masks, pending_parents = [], [], []
            pooled_chunk_outputs = []
            parent_chunk_counts = [0] * video_num_

            def flush_pending_chunks():
                if not pending_videos:
                    return 0
                video = torch.from_numpy(np.stack(pending_videos, axis=0)).to(device)
                video_mask = torch.from_numpy(np.stack(pending_masks, axis=0)).to(device)
                visual_output = model.get_visual_output(video, video_mask)
                # This exactly matches CLIP4Clip's loose meanP visual path.
                flat_mask = video_mask.view(-1, video_mask.shape[-1])
                visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                pooled = (visual_output * flat_mask.to(visual_output.dtype).unsqueeze(-1)).sum(dim=1)
                pooled = pooled / flat_mask.sum(dim=1, keepdim=True).to(visual_output.dtype).clamp_min(1.0)
                pooled_chunk_outputs.append(pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-12))
                parent_indices = torch.tensor(pending_parents, dtype=torch.long, device=device)
                chunk_parent_indices.append(parent_indices)
                count = int(parent_indices.numel())
                pending_videos.clear()
                pending_masks.clear()
                pending_parents.clear()
                return count

            dataset = test_dataloader.dataset
            chunk_batch_size = max(1, int(args.batch_size_val))
            chunk_stats = dataset.prechunk_statistics() or {}
            total_chunks = int(chunk_stats.get("chunks", 0))
            chunk_started = time.time()
            chunk_report_step = max(1, total_chunks // 20)
            chunk_next_report = chunk_report_step
            encoded_chunks = 0
            logger.info("Raw chunk encoding: %d videos, %d chunks, batch_size=%d",
                        video_num_, total_chunks, chunk_batch_size)
            for parent_index, chunk_video, chunk_mask in dataset.iter_prechunked_videos():
                pending_videos.append(chunk_video)
                pending_masks.append(chunk_mask)
                pending_parents.append(parent_index)
                parent_chunk_counts[parent_index] += 1
                if len(pending_videos) >= chunk_batch_size:
                    encoded_chunks += flush_pending_chunks()
                    if encoded_chunks >= chunk_next_report or encoded_chunks == total_chunks:
                        elapsed = time.time() - chunk_started
                        rate = encoded_chunks / max(elapsed, 1e-6)
                        eta = (total_chunks - encoded_chunks) / max(rate, 1e-6)
                        logger.info("Raw chunk encoding: %d/%d (%.1f%%), elapsed %.1fs, ETA %.1fs",
                                    encoded_chunks, total_chunks, 100.0 * encoded_chunks / max(total_chunks, 1),
                                    elapsed, eta)
                        while chunk_next_report <= encoded_chunks:
                            chunk_next_report += chunk_report_step
            encoded_chunks += flush_pending_chunks()
            logger.info("Raw pre-chunk statistics: videos=%d chunks=%d mean_chunks=%.3f max_chunks=%d",
                        len(parent_chunk_counts), encoded_chunks,
                        float(np.mean(parent_chunk_counts)), max(parent_chunk_counts))

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        if raw_prechunked:
            sim_matrix = _prechunked_parent_similarity(
                model, batch_sequence_output_list, pooled_chunk_outputs,
                torch.cat(chunk_parent_indices, dim=0), video_num_, args.batch_size_val,
            )
        elif n_gpu > 1:
            device_ids = list(range(n_gpu))
            batch_list_t_splits = []
            batch_list_v_splits = []
            batch_t_output_splits = []
            batch_v_output_splits = []
            bacth_len = len(batch_list_t)
            split_len = (bacth_len + n_gpu - 1) // n_gpu
            for dev_id in device_ids:
                s_, e_ = dev_id * split_len, (dev_id + 1) * split_len
                if dev_id == 0:
                    batch_list_t_splits.append(batch_list_t[s_:e_])
                    batch_list_v_splits.append(batch_list_v)

                    batch_t_output_splits.append(batch_sequence_output_list[s_:e_])
                    batch_v_output_splits.append(batch_visual_output_list)
                else:
                    devc = torch.device('cuda:{}'.format(str(dev_id)))
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_t[s_:e_]]
                    batch_list_t_splits.append(devc_batch_list)
                    devc_batch_list = [tuple(t.to(devc) for t in b) for b in batch_list_v]
                    batch_list_v_splits.append(devc_batch_list)

                    devc_batch_list = [b.to(devc) for b in batch_sequence_output_list[s_:e_]]
                    batch_t_output_splits.append(devc_batch_list)
                    devc_batch_list = [b.to(devc) for b in batch_visual_output_list]
                    batch_v_output_splits.append(devc_batch_list)

            parameters_tuple_list = [(batch_list_t_splits[dev_id], batch_list_v_splits[dev_id],
                                      batch_t_output_splits[dev_id], batch_v_output_splits[dev_id],
                                      args.chunk_size) for dev_id in device_ids]
            parallel_outputs = parallel_apply(_run_on_single_gpu, model, parameters_tuple_list, device_ids)
            sim_matrix = []
            for idx in range(len(parallel_outputs)):
                sim_matrix += parallel_outputs[idx]
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        else:
            sim_matrix = _run_on_single_gpu(
                model, batch_list_t, batch_list_v, batch_sequence_output_list,
                batch_visual_output_list, args.chunk_size)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

        if args.chunk_size > 0 and not raw_prechunked:
            chunk_counts = []
            for (video_mask,) in batch_list_v:
                video_mask = video_mask.view(-1, video_mask.shape[-1])
                frame_lengths = video_mask.sum(dim=1).detach().cpu().tolist()
                chunk_counts.extend((int(length) + args.chunk_size - 1) // args.chunk_size
                                    for length in frame_lengths if int(length) > 0)
            if chunk_counts:
                logger.info("Chunk statistics: videos=%d mean_chunks=%.3f max_chunks=%d",
                            len(chunk_counts), float(np.mean(chunk_counts)), max(chunk_counts))

    if getattr(args, "multi_gt_eval", False):
        # The score calculation above is intentionally identical to ordinary
        # CLIP4Clip evaluation.  Only now, after ``sim_matrix`` exists, replace
        # the one-positive metric with dense multi-positive coverage recall.
        dataset = test_dataloader.dataset
        query_indices = getattr(dataset, "eval_query_indices", None)
        query_ids = getattr(dataset, "eval_query_ids", None)
        video_ids = getattr(dataset, "eval_video_ids", None)
        gt_path = getattr(dataset, "multi_gt_file", None) or args.multi_gt_file
        if query_indices is None or query_ids is None or video_ids is None or gt_path is None:
            raise ValueError("--multi_gt_eval requires PRVR raw-frame dense metadata")
        metrics = _multi_gt_text_to_video_metrics(sim_matrix[query_indices], query_ids, video_ids, gt_path)
        logger.info("multi-GT sim matrix size: {}, {}; gt file: {}".format(
            len(query_ids), len(video_ids), gt_path))
        logger.info("Text-to-Video multi-GT:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - evaluated queries: {}'.format(
            metrics['R1'], metrics['R5'], metrics['R10'], metrics['evaluated']))
        return metrics['R1']

    if multi_sentence_:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']
        
        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

                ## Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(best_output_model_file, best_score))

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)

if __name__ == "__main__":
    main()
