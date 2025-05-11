CUDA_VISIBLE_DEVICES=2,3 vllm serve /path/critic_m2w_7B --tensor-parallel-size 2 --port 8123
