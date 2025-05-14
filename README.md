# VEM: Environment-Free Exploration for Training GUI Agent with Value Environment Model


We propose an environment-free RL framework that decouples value estimation from policy optimization by leveraging a pretrained Value Environment Model (VEM). VEM predicts state-action values directly from offline data, distilling human-like priors about GUI interaction outcomes without requiring next-state prediction or environmental feedback. The framework operates in two stages: (1) pretraining VEM to estimate long-term action utilities and (2) guiding policy exploration with frozen VEM signals, enabling layout-agnostic GUI automation.

<div align="center">
  <img width="70%" src="docs/structure.jpg">
</div>

## Quick Start 🚀

### Step 1: Build Environment
```bash
conda env create -f environment.yml
conda activate vem

git clone https://github.com/hiyouga/LLaMA-Factory.git 
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### Step 2: Prepare Images and Annotations

In this part we follow [SeeClick](https://github.com/njucckevin/SeeClick)

#### Download Screenshots and Annotations

* Mind2Web: Download the [screenshots](https://box.nju.edu.cn/f/33e203d170ab48b0b922/) and [annotations](https://box.nju.edu.cn/f/e30b861fa7604668821b/) (train set and test set of Domain/Website/Task). 
Note that according to [mind2web](https://github.com/OSU-NLP-Group/Mind2Web), please **DO NOT** redistribute the unzipped data files online.

* AITW: Download the [screenshots](https://box.nju.edu.cn/f/96ba5115bae24eaaa44e/) and [annotations](https://box.nju.edu.cn/f/1245c74fc09b4565a235/) (train/val/test).
Check the origin [AITW](https://github.com/google-research/google-research/tree/master/android_in_the_wild) project for details and data usage.

#### Data Process

- To get the labeled data for training the critic model, fill in the `api_key` and `model_name` in `configs/gpt_config.yaml`.
- For AITW, run `python3 data_preprocess/aitw.py` to generate the data for training the critic model and policy model. For Mind2Web, run `m2w_process_data.py`

### Step 3: Prepare Checkpoints

Download the checkpoints from:
- [Auto-UI-Base](https://huggingface.co/cooelf/Auto-UI/tree/main) (choose the base version)
- [BLIP2-OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b)
- [RoBERTa-Base](https://huggingface.co/FacebookAI/roberta-base)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

### Step 4: Train the Critic Model
We use the LLaMA-Factory to train the critic model. Based on our settings, this requires 8 A100 GPUs and uses LoRA for training.

#### AITW 

- To obtain the critic model checkpoint for the AITW general task, run:
  ```bash
  sh scripts/train_critic_general.sh
  ```
  The critic checkpoints will be stored in `checkpoints/critic_general`.
- To obtain the critic model checkpoint for the AITW webshopping task, run:
  ```bash
  sh scripts/train_critic_webshopping.sh
  ```
  The critic checkpoints will be stored in `checkpoints/critic_webshopping`.

#### MM-Mind2Web

To obtain the critic model checkpoint for the MM-Mind2Web, run:

```
sh scripts/train_critic.sh
```

You can modify the output path by changing the `output_dir` in the YAML file. Remember to fill in the `adapter_name_or_path` and `export_dir` in `configs/critic_merge.yaml` when merging LoRA.

### Step 5: Train the Policy Model

#### AITW

After obtaining the critic model, we use AutoGUI as the base policy model for training:
```bash
python3 train.py --task general
python3 train.py --task webshopping
```
Checkpoints are saved in `checkpoints/policy_general` and `checkpoints/policy_webshopping` by default.

#### MM-Mind2Web

Use VLLM to run the critic model as a API server:

```bash
bash run_vllm_critic.sh
```

Then run the policy model training:

```bash
python3 train.py
```

### Step 6: Evaluation

#### AITW

- **Offline Evaluation**
Please modify the save_path to point to the exact checkpoints you want to evaluate (which you obtained in Step 5).
  ```bash
  python3 train.py --task general --eval
  python3 train.py --task webshopping --eval
  ```
- **Online Evaluation**
  - Set up the Android environment according to this [page](https://github.com/DigiRL-agent/digirl/tree/master/env_setup), obtain the URL, and fill in the `appium_server_url` in `configs/online_eval.yaml`.
  - Run the agent demo using:
    ```bash
    python3 models/demo.py --model_path xxx
    ```
    Obtain the Gradio public URL and fill in the `agent_url` in `configs/online_eval_general.yaml` or `configs/online_eval_webshopping.yaml`.
  - Execute:
    ```bash
    python3 eval_online.py --task general
    python3 eval_online.py --task webshopping
    ```

#### MM-Mind2Web

We use the same evaluation code as [SeeClick](https://github.com/njucckevin/SeeClick), see [mind2web_test.py](https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/mind2web_test.py), run the MM-Mind2Web evaluation by `python test_mind2web.py`.