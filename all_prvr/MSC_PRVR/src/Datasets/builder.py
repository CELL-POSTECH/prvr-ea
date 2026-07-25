import os
import ipdb
from torch.utils.data import DataLoader

from Utils.basic_utils import BigFile, read_dict
from prvr_compat import text_feature_path
from Datasets.data_provider import Dataset4PRVR, VisDataSet4PRVR, TxtDataSet4PRVR, \
    collate_train, collate_frame_val, collate_text_val, read_video_ids


def get_datasets(cfg):
    rootpath = cfg['data_root']
    annotation_root = os.path.join(rootpath, 'annotations')
    collection = cfg['collection']
    trainCollection = '%strain' % collection
    valCollection = '%sval' % collection
    cap_file = {
        'train': '%s.caption.txt' % trainCollection,
        'val': '%s.caption.txt' % valCollection,
    }
    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x]) for x in cap_file}
    multi_gt_caption_file = os.environ.get('PRVR_MULTI_GT_CAPTION_FILE')
    use_tvr_multigt_eval = bool(os.environ.get('PRVR_MULTI_GT')) and collection == 'tvr'
    if use_tvr_multigt_eval:
        caption_files['val'] = multi_gt_caption_file or os.path.join(rootpath, collection, 'tvrdenseval_v.caption.txt')

    # Load visual features
    if cfg['feature_mode'] == 'clip':
        visual_feats = os.path.join(rootpath, collection, 'FeatureData', 'new_clip_vit_32_%s_vid_features.hdf5' % collection)
        text_feat_path = os.path.join(rootpath, collection, 'TextData', 'clip_ViT_B_32_%s_query_feat.hdf5' % collection)
        video2frames = None
        test_visual_feats = visual_feats
        test_video2frames = None
        is_clip = True

    elif cfg['visual_feature'] == 'clipsf': #qvhighlights
        visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_slowfast_clip.h5' % collection)
        text_feat_path = os.path.join(rootpath, collection, 'TextData', '%s_slowfast_clip_text.h5' % collection)
        cfg['visual_feat_dim'] = 2816
        cfg['q_feat_size'] = 512
        video2frames = None
        test_visual_feats = os.path.join(rootpath, collection, 'FeatureData', '%s_slowfast_clip.h5' % collection)
        test_video2frames = None
        is_clip = True

    else:
        visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
        visual_feats = BigFile(visual_feat_path)
        cfg['visual_feat_dim'] = visual_feats.ndims

        text_feat_path = text_feature_path(rootpath, collection, cfg['feature_mode'])
        video2frames = read_dict(
            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))

        test_visual_feat_path = os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'])
        test_visual_feats = BigFile(test_visual_feat_path)
        test_video2frames = read_dict(
            os.path.join(rootpath, collection, 'FeatureData', cfg['visual_feature'], 'video2frames.txt'))
        is_clip = False

    if collection == 'tvr':
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json=os.path.join(annotation_root, 'tvr_train_release.jsonl'))

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json=None if use_tvr_multigt_eval else os.path.join(annotation_root, 'tvr_val_release.jsonl'))

    elif collection in ('activitynet', 'msrvtt'):
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json=None)

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json=None)
    elif collection == 'qvhighlight':
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json='/project/prvr/dataset/qvhighlight/highlight_train_release.jsonl')

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json='/project/prvr/dataset/qvhighlight/highlight_val_release.jsonl')
    elif collection == 'charades':
        train_dataset = Dataset4PRVR(caption_files['train'], visual_feats, text_feat_path, cfg,
                                     video2frames=video2frames, is_clip=is_clip,
                                     path_query_json=None)

        val_text_dataset = TxtDataSet4PRVR(caption_files['val'], text_feat_path, cfg,
                                           path_query_json=None)
    else:
        raise ValueError("inapposite collection")

    val_video_ids_list = read_video_ids(caption_files['val'])
    val_video_dataset = VisDataSet4PRVR(visual_feats, video2frames, cfg, video_ids=val_video_ids_list, is_clip=is_clip)

    testCollection = '%stest' % collection
    test_cap_file = {'test': '%s.caption.txt' % testCollection}

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg['batchsize'],
                              shuffle=True,
                              pin_memory=cfg['pin_memory'],
                              num_workers=cfg['num_workers'],
                              collate_fn=collate_train)
    context_dataloader = DataLoader(val_video_dataset,
                                    collate_fn=collate_frame_val,
                                    batch_size=cfg['eval_context_bsz'],
                                    num_workers=cfg['num_workers'],
                                    shuffle=False,
                                    pin_memory=cfg['pin_memory'])
    query_eval_loader = DataLoader(val_text_dataset,
                                   collate_fn=collate_text_val,
                                   batch_size=cfg['eval_query_bsz'],
                                   num_workers=cfg['num_workers'],
                                   shuffle=False,
                                   pin_memory=cfg['pin_memory'])
    return cfg, train_loader, context_dataloader, query_eval_loader
