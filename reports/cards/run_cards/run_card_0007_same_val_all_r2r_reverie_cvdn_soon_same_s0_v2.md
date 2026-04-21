# Run Card

## Basic
- experiment_id: 0007_same_val_all_r2r_reverie_cvdn_soon_same_s0_v2
- date: 2026-04-21 10:00:18+08:00 (finished 2026-04-21 10:30:51+08:00)
- run_type: checkpoint_eval
- project_repo: vln-lab (/workspace/vln-lab)
- project_commit: db3552571a15f38ccabd768373570cdb27e49d5d
- submodule_repo: third_party/SAME
- submodule_commit: b74c57bbb189a0ed8c908e90b53d02dfbe417528
- command: /opt/conda/envs/test-v1/bin/python run.py --config_dir /workspace/vln-lab/configs/same/val_all_r2r_reverie_cvdn_soon.yaml --options experiment.id=0007_same_val_all_r2r_reverie_cvdn_soon_same_s0_v2 experiment.output_dir=../../../experiment_outputs
- config_file: configs/same/val_all_r2r_reverie_cvdn_soon.yaml
- config_hash: sha256:65efdfe7dc5c0144bcfb9150bd386bc9d87e7a21a7fcb555721c0e3ad6215f15

## Environment
- machine: xzy--Y9000P
- GPU: 1 x NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB)
- driver: 580.126.09
- CUDA: driver 13.0 / torch 12.8
- torch: 2.7.0+cu128
- conda_env / docker_image: test-v1 /

## Data
- data_root: /workspace/vln-lab/data/same
- dataset_files: R2R/R2R_val_train_seen_enc.json; R2R/R2R_val_seen_enc.json; R2R/R2R_val_unseen_enc.json; REVERIE/REVERIE_val_train_seen_enc.json; REVERIE/REVERIE_val_seen_enc.json; REVERIE/REVERIE_val_unseen_enc.json; REVERIE/BBoxes.json; CVDN/val_seen.json; CVDN/val_unseen.json; SOON/val_unseen_instrs_enc_pseudo_obj_ade30k_label.jsonl; SOON/val_unseen_house_enc_pseudo_obj_ade30k_label.jsonl
- split: R2R: val_train_seen, val_seen, val_unseen; REVERIE: val_train_seen, val_seen, val_unseen; CVDN: val_seen, val_unseen; SOON: val_seen, val_unseen
- sample_count: R2R val_train_seen=1501; R2R val_seen=1021; R2R val_unseen=2349; REVERIE val_train_seen=123; REVERIE val_seen=1423; REVERIE val_unseen=3521; CVDN val_seen=382; CVDN val_unseen=907; SOON val_seen=1130; SOON val_unseen=3390; total=15747
- data_manifest: experiment_outputs/0007_same_val_all_r2r_reverie_cvdn_soon_same_s0_v2/data_manifest.txt

## Model
- checkpoint: ../../../data/same/ckpt/SAME.pt
- checkpoint_hash: sha256:b62925399b658a106556168ec09c8c8e594ec485ee3238c5e18c3adbeb8c4662
- architecture: duet + GlocalTextPathNavCMT; Task MoE at Attn_q; 8 experts, top-2 routing
- parameter_count: 215.74 M trainable
- changed_modules: third_party/SAME/src/configs/default.yaml; third_party/SAME/src/run.py; third_party/SAME/src/tasks/datasets/cvdn.py; third_party/SAME/src/tasks/datasets/soon.py; third_party/SAME/src/utils/common_utils.py; configs/same/val_all_r2r_reverie_cvdn_soon.yaml

## Protocol
- eval_or_train: eval (eval_only=true)
- seed: 0
- batch_size: 8
- val_batch_size: 4
- decoding: argmax
- max_action_len: R2R=15; REVERIE=15; CVDN=30; SOON=20

## Status
- success / partial / failed: success
- output_files: run.json; metrics.json; config_resolved.yaml; patch.diff; stdout.log; stderr.log; results/R2R_val_train_seen_results.json; results/R2R_val_seen_results.json; results/R2R_val_unseen_results.json; results/REVERIE_val_train_seen_results.json; results/REVERIE_val_seen_results.json; results/REVERIE_val_unseen_results.json; results/CVDN_val_seen_results.json; results/CVDN_val_unseen_results.json; results/SOON_val_seen_results.json; results/SOON_val_unseen_results.json
- known_issues: REVERIE 的 `rgs/rgspl` 与 SOON 的 `det_sr/det_spl` 属于默认 DUET runtime 下未激活的 object grounding / detection 扩展字段，不能作为论文主结果对照；official_results.csv 只覆盖少量 headline 指标，因此多数 split 和诊断指标没有官方参考；本次运行使用 dirty worktree，并包含 CVDN/SOON path metrics 的实验性 evaluator 改动；data_manifest 中若干 habitat/object 路径标记 missing，但本次 mattersim + feature.enable_og=false 的评估流程未触发

## Notes
- `conda_env / docker_image` 中的 `docker_image` 留空，因为归档里没有镜像名记录。
- `config_hash` 基于归档后的 `config_resolved.yaml` 计算，不是直接对原始配置文件做哈希。
- `patch_set` 只记录了 base patches；CVDN/SOON 的 nDTW、SDTW、CLS 来自运行时 worktree 中的实验性 evaluator 修改，具体差异见 `patch.diff`。
