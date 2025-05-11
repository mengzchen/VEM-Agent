#!/usr/bin/env bash

llamafactory-cli train configs/train_critic_m2w.yaml

# fill the adapter_name_or_path and export_dir in critic_merge.yaml
llamafactory-cli export configs/critic_merge_m2w.yaml