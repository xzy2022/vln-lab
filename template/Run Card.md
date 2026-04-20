# Run Card

## Basic
- experiment_id:
- date:
- run_type: smoke / checkpoint_eval / train / ablation / diagnostic
- project_repo:
- project_commit:
- submodule_repo:
- submodule_commit:
- command:
- config_file:
- config_hash:

## Environment
- machine:
- GPU:
- driver:
- CUDA:
- torch:
- conda_env / docker_image:

## Data
- data_root:
- dataset_files:
- split:
- sample_count:
- data_manifest:

## Model
- checkpoint:
- checkpoint_hash:
- architecture:
- parameter_count:
- changed_modules:

## Protocol
- eval_or_train:
- seed:
- batch_size:
- val_batch_size:
- decoding:
- max_action_len:

## Status
- success / partial / failed:
- output_files:
- known_issues: