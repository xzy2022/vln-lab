# Run Card

## Basic
- experiment_id: 0005_same_val_r2r_reverie_cvdn_soon_same_s0_v1
- date: 2026-04-20 08:40:06+00:00 (finished 2026-04-20 09:02:14+00:00)
- run_type: checkpoint_eval
- project_repo: vln-lab (/workspace/vln-lab)
- project_commit: 5a7ccf7367e310b3205a7136bcaec0a74efb7dc0
- submodule_repo: third_party/SAME
- submodule_commit: b74c57bbb189a0ed8c908e90b53d02dfbe417528
- command: /opt/conda/envs/test-v1/bin/python run.py --config_dir /workspace/vln-lab/configs/same/val_r2r_reverie_cvdn_soon.yaml --options experiment.id=0005_same_val_r2r_reverie_cvdn_soon_same_s0_v1 experiment.output_dir=../../../experiment_outputs
- config_file: configs/same/val_r2r_reverie_cvdn_soon.yaml
- config_hash: sha256:59e21a140776e57c4c63befbc8cea005f9ac7c503b760d9c575b1e688a759395

## Environment
- machine: xzy--Y9000P
- GPU: 1 x NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB)
- driver: 580.126.09
- CUDA: driver 13.0 / torch 12.8
- torch: 2.7.0+cu128
- conda_env / docker_image: test-v1 /

## Data
- data_root: /workspace/vln-lab/data/same
- dataset_files: R2R/R2R_val_train_seen_enc.json; R2R/R2R_val_unseen_enc.json; REVERIE/REVERIE_val_train_seen_enc.json; REVERIE/REVERIE_val_unseen_enc.json; REVERIE/BBoxes.json; CVDN/val_unseen.json; SOON/val_unseen_house_enc_pseudo_obj_ade30k_label.jsonl
- split: CVDN: val_unseen; R2R: val_train_seen, val_unseen; REVERIE: val_train_seen, val_unseen; SOON: val_unseen
- sample_count: R2R val_train_seen=1501; R2R val_unseen=2349; REVERIE val_train_seen=123; REVERIE val_unseen=3521; CVDN val_unseen=907; SOON val_unseen=3390; total=11791
- data_manifest: experiment_outputs/0005_same_val_r2r_reverie_cvdn_soon_same_s0_v1/data_manifest.txt

## Model
- checkpoint: ../../../data/same/ckpt/SAME.pt
- checkpoint_hash: sha256:b62925399b658a106556168ec09c8c8e594ec485ee3238c5e18c3adbeb8c4662
- architecture: duet + GlocalTextPathNavCMT; Task MoE at Attn_q; 8 experts, top-2 routing
- parameter_count: 215.74 M trainable
- changed_modules: scripts/experiments/run_same.py; third_party/SAME/src/configs/default.yaml; third_party/SAME/src/run.py; third_party/SAME/src/utils/common_utils.py

## Protocol
- eval_or_train: eval (eval_only=true)
- seed: 0
- batch_size: 8
- val_batch_size: 4
- decoding: argmax
- max_action_len: R2R=15; REVERIE=15; CVDN=30; SOON=20

## Status
- success / partial / failed: success
- output_files: run.json; metrics.json; config_resolved.yaml; patch.diff; stdout.log; stderr.log; results/R2R_val_train_seen_results.json; results/R2R_val_unseen_results.json; results/REVERIE_val_train_seen_results.json; results/REVERIE_val_unseen_results.json; results/CVDN_val_unseen_results.json; results/SOON_val_unseen_results.json
- known_issues: official_results.csv 缺少多数参考项，论文对比只能覆盖少量指标; 运行时父仓库与 third_party/SAME 均为 dirty worktree，精确差异需结合 patch.diff; data_manifest 中若干 habitat/object 路径标记 missing，但本次 mattersim + feature.enable_og=false 的评估流程未触发

## Notes
- `conda_env / docker_image` 中的 `docker_image` 留空，因为归档里没有镜像名记录。
- `config_hash` 基于归档后的 `config_resolved.yaml` 计算，不是直接对原始配置文件做哈希。
