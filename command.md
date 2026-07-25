# PRVR 재현 명령

```bash
export PRVR_PROJECT_ROOT=/path/to/prvr-reproduction
export PRVR_DATA_ROOT="$PRVR_PROJECT_ROOT/datasets"
export PYTHONPATH="$PRVR_PROJECT_ROOT/all_prvr${PYTHONPATH:+:$PYTHONPATH}"
```

`<GPU>`는 사용할 GPU 번호다. `resnet`은 각 원본 모델의 I3D/ResNet visual
feature와 RoBERTa/BERT text feature를 뜻하며, `clip`은 제공된 CLIP ViT-B/32
visual/text feature를 student 입력으로 사용한다. 결과는 모두 각 프로젝트의
`results/<resnet|clip>/<dataset>/...` 아래에 저장된다.

## Config 기반 모델

아래 명령은 원본 dataset config의 hyperparameter를 그대로 사용한다. `*_clip`은
feature reader와 입력 차원만 CLIP 512-D로 바꾼 조건이며, `msrvtt`는 ActivityNet
config를 기반으로 추가한 dataset 조건이다.

### AMDNet

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/AMDNet"
python main.py -d act --gpu <GPU>
python main.py -d act_clip --gpu <GPU>
python main.py -d tvr --gpu <GPU>
python main.py -d tvr_clip --gpu <GPU>
python main.py -d cha --gpu <GPU>
python main.py -d cha_clip --gpu <GPU>
python main.py -d msrvtt --gpu <GPU>
python main.py -d msrvtt_clip --gpu <GPU>
```

### BOA

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/BOA"
python src/main.py -d act --gpu <GPU>
python src/main.py -d act_clip --gpu <GPU>
python src/main.py -d tvr --gpu <GPU>
python src/main.py -d tvr_clip --gpu <GPU>
python src/main.py -d cha --gpu <GPU>
python src/main.py -d cha_clip --gpu <GPU>
python src/main.py -d msrvtt --gpu <GPU>
python src/main.py -d msrvtt_clip --gpu <GPU>
```

### DreamPRVR

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/CVPR26-DreamPRVR/src"
python main.py -d act --gpu <GPU>
python main.py -d act_clip --gpu <GPU>
python main.py -d tvr --gpu <GPU>
python main.py -d tvr_clip --gpu <GPU>
python main.py -d cha --gpu <GPU>
python main.py -d cha_clip --gpu <GPU>
python main.py -d msrvtt --gpu <GPU>
python main.py -d msrvtt_clip --gpu <GPU>
```

### GMMFormer

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/GMMFormer/src"
python main.py -d act --gpu <GPU>
python main.py -d act_clip --gpu <GPU>
python main.py -d tvr --gpu <GPU>
python main.py -d tvr_clip --gpu <GPU>
python main.py -d cha --gpu <GPU>
python main.py -d cha_clip --gpu <GPU>
python main.py -d msrvtt --gpu <GPU>
python main.py -d msrvtt_clip --gpu <GPU>
```

### GMMFormer v2

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/GMMFormer_v2/src"
python main.py -d act --gpu <GPU>
python main.py -d act_clip --gpu <GPU>
python main.py -d tvr --gpu <GPU>
python main.py -d tvr_clip --gpu <GPU>
python main.py -d cha --gpu <GPU>
python main.py -d cha_clip --gpu <GPU>
python main.py -d msrvtt --gpu <GPU>
python main.py -d msrvtt_clip --gpu <GPU>
```

### HLFormer

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/ICCV25-HLFormer/src"
python main.py -d act --gpu <GPU>
python main.py -d act_clip --gpu <GPU>
python main.py -d tvr --gpu <GPU>
python main.py -d tvr_clip --gpu <GPU>
python main.py -d cha --gpu <GPU>
python main.py -d cha_clip --gpu <GPU>
python main.py -d msrvtt --gpu <GPU>
python main.py -d msrvtt_clip --gpu <GPU>
```

### Holmes

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/ICML26-Holmes/src"
python main.py -d act --gpu <GPU>
python main.py -d act_clip --gpu <GPU>
python main.py -d tvr --gpu <GPU>
python main.py -d tvr_clip --gpu <GPU>
python main.py -d cha --gpu <GPU>
python main.py -d cha_clip --gpu <GPU>
python main.py -d msrvtt --gpu <GPU>
python main.py -d msrvtt_clip --gpu <GPU>
```

### MSC-PRVR

MSC-PRVR은 요청대로 CLIP feature 조건만 제공한다.

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/MSC_PRVR/src"
python main.py -d act_clip --gpu <GPU>
python main.py -d tvr_clip --gpu <GPU>
python main.py -d cha_clip --gpu <GPU>
python main.py -d msrvtt_clip --gpu <GPU>
```

Config 기반 모델의 평가는 학습과 같은 directory에서 다음처럼 실행한다.

```bash
python main.py -d <DATASET> --gpu <GPU> --eval --resume <ABSOLUTE_BEST_CKPT>
```

## BGM-Net

원본 `do_*.sh`의 dataset별 설정을 명시했다. TVR의 text dimension은 ResNet일 때
768-D, CLIP일 때 512-D로 feature file에서 자동 선택된다.

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/BGM-Net"

# ActivityNet: original do_activitynet.sh
python method/train.py --collection activitynet --dset_name activitynet --root_path "$PRVR_DATA_ROOT" --visual_feature i3d --feature_mode resnet --output_root "$PWD/results" --exp_id act_resnet --device_ids <GPU> --use_matcher_start_epoch 20 --map_size 48 --smp_rate 1.0 --bsz 128
python method/train.py --collection activitynet --dset_name activitynet --root_path "$PRVR_DATA_ROOT" --visual_feature i3d --feature_mode clip --output_root "$PWD/results" --exp_id act_clip --device_ids <GPU> --use_matcher_start_epoch 20 --map_size 48 --smp_rate 1.0 --bsz 128

# TVR: original do_tvr.sh
python method/train.py --collection tvr --dset_name tvr --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_resnet --feature_mode resnet --output_root "$PWD/results" --exp_id tvr_resnet --device_ids <GPU> --margin 0.1 --bsz 128 --lr 0.00025 --use_matcher_start_epoch 0 --map_size 32 --smp_rate 0.01
python method/train.py --collection tvr --dset_name tvr --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_resnet --feature_mode clip --output_root "$PWD/results" --exp_id tvr_clip --device_ids <GPU> --margin 0.1 --bsz 128 --lr 0.00025 --use_matcher_start_epoch 0 --map_size 32 --smp_rate 0.01

# Charades: original do_charades.sh
python method/train.py --collection charades --dset_name charades --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_rgb_lgi --feature_mode resnet --output_root "$PWD/results" --exp_id cha_resnet --device_ids <GPU> --clip_scale_w 0.6 --frame_scale_w 0.4 --use_matcher_start_epoch 5 --map_size 48 --smp_rate 1.0 --bsz 16
python method/train.py --collection charades --dset_name charades --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_rgb_lgi --feature_mode clip --output_root "$PWD/results" --exp_id cha_clip --device_ids <GPU> --clip_scale_w 0.6 --frame_scale_w 0.4 --use_matcher_start_epoch 5 --map_size 48 --smp_rate 1.0 --bsz 16

# MSRVTT: ActivityNet original setting을 새 dataset에 적용
python method/train.py --collection msrvtt --dset_name msrvtt --root_path "$PRVR_DATA_ROOT" --visual_feature resnext101-resnet152 --feature_mode resnet --output_root "$PWD/results" --exp_id msrvtt_resnet --device_ids <GPU> --use_matcher_start_epoch 20 --map_size 48 --smp_rate 1.0 --bsz 128
python method/train.py --collection msrvtt --dset_name msrvtt --root_path "$PRVR_DATA_ROOT" --visual_feature resnext101-resnet152 --feature_mode clip --output_root "$PWD/results" --exp_id msrvtt_clip --device_ids <GPU> --use_matcher_start_epoch 20 --map_size 48 --smp_rate 1.0 --bsz 128
```

## MS-SL

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/ms-sl"

# ActivityNet: original do_activitynet.sh
python method/train.py --collection activitynet --dset_name activitynet --root_path "$PRVR_DATA_ROOT" --visual_feature i3d --feature_mode resnet --output_root "$PWD/results" --exp_id act_resnet --device_ids <GPU>
python method/train.py --collection activitynet --dset_name activitynet --root_path "$PRVR_DATA_ROOT" --visual_feature i3d --feature_mode clip --output_root "$PWD/results" --exp_id act_clip --device_ids <GPU>

# TVR: original do_tvr.sh
python method/train.py --collection tvr --dset_name tvr --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_resnet --feature_mode resnet --output_root "$PWD/results" --exp_id tvr_resnet --device_ids <GPU> --margin 0.1
python method/train.py --collection tvr --dset_name tvr --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_resnet --feature_mode clip --output_root "$PWD/results" --exp_id tvr_clip --device_ids <GPU> --margin 0.1

# Charades: original do_charades.sh
python method/train.py --collection charades --dset_name charades --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_rgb_lgi --feature_mode resnet --output_root "$PWD/results" --exp_id cha_resnet --device_ids <GPU> --clip_scale_w 0.5 --frame_scale_w 0.5
python method/train.py --collection charades --dset_name charades --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_rgb_lgi --feature_mode clip --output_root "$PWD/results" --exp_id cha_clip --device_ids <GPU> --clip_scale_w 0.5 --frame_scale_w 0.5

# MSRVTT: ActivityNet original setting을 새 dataset에 적용
python method/train.py --collection msrvtt --dset_name msrvtt --root_path "$PRVR_DATA_ROOT" --visual_feature resnext101-resnet152 --feature_mode resnet --output_root "$PWD/results" --exp_id msrvtt_resnet --device_ids <GPU>
python method/train.py --collection msrvtt --dset_name msrvtt --root_path "$PRVR_DATA_ROOT" --visual_feature resnext101-resnet152 --feature_mode clip --output_root "$PWD/results" --exp_id msrvtt_clip --device_ids <GPU>
```

BGM-Net/MS-SL은 학습 종료 후 test evaluation을 자동으로 실행한다. 별도 평가는
학습이 생성한 absolute run directory를 사용한다.

```bash
python method/eval.py --collection <COLLECTION> --root_path "$PRVR_DATA_ROOT" --visual_feature <VISUAL_FEATURE> --feature_mode <resnet|clip> --model_dir <ABSOLUTE_RUN_DIR> --device_ids <GPU>
```

## DL-DKD

DL-DKD는 원본부터 dual-feature 구조다. student는 I3D/ResNet visual +
RoBERTa/BERT text feature를, teacher는 CLIP-B/32 visual/text feature를 동시에
사용한다. `train_clip.sh`/`eval_clip.sh`에서도 이 원본 dual-feature 설정을
그대로 사용하며, CLIP feature가 teacher로 포함된 결과를 `clip` 조건으로
기록한다. 별도의 CLIP-student 변형은 만들지 않는다.

```bash
cd "$PRVR_PROJECT_ROOT/all_prvr/DL-DKD"

# ActivityNet: original do_activitynet.sh
python method/train.py --collection activitynet --dset_name activitynet --root_path "$PRVR_DATA_ROOT" --visual_feature i3d --results_root "$PWD/results/resnet" --exp_id act_resnet --model_name DLDKD --device_ids <GPU> --distill_loss_decay exp --double_branch --drop 0.25 --input_drop 0.25 --q_feat_size 1024 --label_style soft

# TVR: original do_tvr.sh
python method/train.py --collection tvr --dset_name tvr --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_resnet --results_root "$PWD/results/resnet" --exp_id tvr_resnet --model_name DLDKD --device_ids <GPU> --q_feat_size 768 --margin 0.1 --n_heads 4 --lr 0.0003 --distill_loss_decay exp --double_branch --drop 0.2 --input_drop 0.2 --label_style soft

# Charades: original do_charades.sh
python method/train.py --collection charades --dset_name charades --root_path "$PRVR_DATA_ROOT" --visual_feature i3d_rgb_lgi --results_root "$PWD/results/resnet" --exp_id cha_resnet --model_name DLDKD --device_ids <GPU> --lr 0.00024 --distill_loss_decay exp --double_branch --q_feat_size 1024 --drop 0.15 --input_drop 0.15 --label_style soft

# MSRVTT: ActivityNet original setting을 새 dataset에 적용
python method/train.py --collection msrvtt --dset_name msrvtt --root_path "$PRVR_DATA_ROOT" --visual_feature resnext101-resnet152 --results_root "$PWD/results/resnet" --exp_id msrvtt_resnet --model_name DLDKD --device_ids <GPU> --distill_loss_decay exp --double_branch --drop 0.25 --input_drop 0.25 --q_feat_size 1024 --label_style soft
```

DL-DKD 평가에서 `<RUN_DIR>`은 `results/` 아래의 상대 경로다.

```bash
RUN_DIR=resnet/<DATASET>/<DATASET>-<EXP_ID>_<TIMESTAMP>
python method/eval.py --collection <COLLECTION> --root_path "$PRVR_DATA_ROOT" --visual_feature <VISUAL_FEATURE> --model_dir "$RUN_DIR" --device_ids <GPU>
```
