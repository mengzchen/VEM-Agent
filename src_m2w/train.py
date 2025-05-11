import argparse
import yaml
import os
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from qwen_vl_utils import process_vision_info
import logging
import random
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

import utils
from dataset import ReplayBuffer, DummyDataset
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from agent import Agent
from io import BytesIO
import base64
from typing import List, Dict
import json
from utils import update_trajectory, extract_answer, get_first_number_or_default
from prompt import action_prompt
import PIL
from openai import OpenAI


class QwenPolicyTrainer:
    def __init__(self, agent, accelerator, config):
        self.agent = agent
        self.accelerator = accelerator
        self.grad_accum_steps = config["grad_accum_steps"]
        self.max_grad_norm = config["max_grad_norm"]
        self.gamma = config["gamma"]
        self.epochs = config["epochs"]
        self.tau = config["tau"]
        self.step = 0
        self.lm_optimizer = torch.optim.Adam(agent.model.parameters(), lr=float(config["lm_lr"]))

    def prepare(self):
        self.lm_optimizer = self.accelerator.prepare(self.lm_optimizer)

    def actor_loss(
        self,
        critic_images,
        critic_inputs,
        policy_inputs,
        policy_outputs,
        policy_images,
        validation=False,
        **kwargs
    ):
        # print(f"len of critic inputs: {len(critic_inputs)}")
        # print(f"len of policy inputs: {len(policy_inputs)}")
        
        # 1. Critic part
        dtype = self.accelerator.unwrap_model(self.agent.model).dtype
        device = self.accelerator.unwrap_model(self.agent.model).device

        # Use OpenAI API to get q_values
        client = OpenAI(
            base_url="http://127.0.0.1:8123/v1",
            api_key="test"
        )
        
        q_values = []
        
        for critic_input, critic_image in zip(critic_inputs, critic_images):
            # print(f"### critic_input: {critic_input}")
            
            # Process image to base64 encoding
            if isinstance(critic_image, str):
                # If it's an image path
                with open(critic_image, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            else:
                # If it's a PIL Image object
                buffered = BytesIO()
                critic_image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model="/model/path/to/your/model",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": critic_input},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=128
            )
            
            # Extract q_value
            q_value = response.choices[0].message.content
            q_value = get_first_number_or_default(q_value)
            # print(f"### API q_value response: {q_value}")
            
            q_values.append(q_value)
            
        
        # Convert to tensor and normalize
        q_values = torch.tensor(q_values, dtype=dtype, requires_grad=True).to(device)
        q_values = q_values / 2  # Normalize

        # 2. Policy part
        # Ensure policy_images are PIL.Image
        policy_images = [
            Image.open(img) if isinstance(img, str) else img
            for img in policy_images
        ]

        real_policy_inputs = []
        for task, actions in zip(policy_inputs['task'], policy_inputs['previous_actions']):
            real_policy_inputs.append(action_prompt.replace("{{task}}", task).replace("{{action_history}}", actions))
        # Calculate log probability
        log_prob = self.agent.get_log_prob(
            texts=real_policy_inputs,
            images=policy_images,
            targets=policy_outputs
        )
        

        # print(f"### log_prob: {log_prob}")
        # print(f"### q_values: {q_values}")
        # If log_prob is a list, convert it to tensor
        if isinstance(log_prob, list):
            log_prob = torch.tensor(log_prob, dtype=dtype, device=device)
            
        # Check log_prob dimensions and handle appropriately
        # if len(log_prob.shape) == 1:
        #     # If log_prob has only one dimension, no need for sum operation
        #     pass
        # else:
        #     # Otherwise, sum along the second dimension
        #     log_prob = log_prob.sum(dim=1)

        # print(f"### q_values: {q_values}")
        # print(f"### log_prob shape: {log_prob.shape}, q_values shape: {q_values.shape}")

        # 3. Policy Gradient loss
        pg_loss = - torch.mean(log_prob * q_values) / 100.0   # set your own scale

        # print(f"### pg_loss: {pg_loss}")

        if not validation:
            self.accelerator.backward(pg_loss)

        return pg_loss.detach().cpu().item(), torch.mean(q_values).detach().cpu().item()

    def update_policy(self, buffer, is_validation, batch_size):
        logs = []

        self.step += 1
        data = [buffer.sample(1) for _ in range(self.grad_accum_steps * batch_size)]

        for d in data:
            for k, v in d.items():
                d[k] = v[0]

        # print(f"### data.keys(): {data[0].keys()}")
        # input("stop here to see the data")

        keys = ["ep_id", "step_id", "policy_inputs", "policy_outputs", "policy_images", "critic_inputs", "critic_images"]
        dataloader = self.accelerator.prepare(
            DataLoader(
                DummyDataset(data, keys), 
                batch_size=batch_size, 
                shuffle=False,
            )
        )

        self.lm_optimizer.zero_grad()
        losses, q_values = [], []


        if is_validation:
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Validation Batches", leave=False, disable=not self.accelerator.is_main_process):
                    loss, q_value = self.actor_loss(**batch, validation=True)
                    losses.append(loss)
                    q_values.append(q_value)
            logging.info(f"[val] step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "val loss": sum(losses) / len(losses), "val Q value": sum(q_values) / len(q_values)})
        else:
            for batch in tqdm(dataloader, desc="Training Batches", leave=False, disable=not self.accelerator.is_main_process):
                loss, q_value = self.actor_loss(**batch, validation=False)
                losses.append(loss)
                q_values.append(q_value)
            logging.info(f"step: {self.step}\tloss: {sum(losses) / len(losses):.2f}\tQ-values: {sum(q_values) / len(q_values):.4f}")
            logs.append({"step": self.step, "train loss": sum(losses) / len(losses), "train Q value": sum(q_values) / len(q_values)})

            self.accelerator.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.lm_optimizer.step()

        return logs

    def infer(self, data, batch_size):
        """
        Sample Policy Model outputs, adding "output", "groundtruth", "ep_id", "step_id" fields to data
        """
        # "policy_input", "policy_output", "policy_image" are generated by process_m2w_data.py get_rl_data()
        keys = ["ep_id", "step_id", "policy_input", "policy_output", "policy_image"]
        dataloader = DataLoader(
            DummyDataset(data, keys), 
            batch_size=batch_size, 
            shuffle=False,
            # collate_fn=custom_collate_fn
        )
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        for batch in tqdm(dataloader, desc="Inference", leave=False, disable=not self.accelerator.is_main_process):
            ep_ids, step_ids = batch["ep_id"], batch["step_id"]

            policy_inputs, groundtruths, image_paths = batch["policy_input"], batch["policy_output"], batch["policy_image"]

            texts = []

            # for policy_input in policy_inputs:
            for task, actions in zip(policy_inputs['task'], policy_inputs['previous_actions']):
                # policy_input: {
                #     "task": ann["task"],
                #     "previous_actions": previous_actions_str
                # }
                # print(f"### task: {task}")
                # print(f"### actions: {actions}")
                # input("stop here")
                action_prompt_input = action_prompt.replace("{{task}}", task).replace("{{action_history}}", actions)
                texts.append(action_prompt_input)
            
            # print(f"### texts: {texts}")
            # print(f"### image_paths: {image_paths}")
            
            outputs = self.agent.get_action(texts, image_paths)
            # print(f"### outputs: {outputs}")
            # input("stop here")

            assert len(outputs) == len(groundtruths) == len(ep_ids) == len(step_ids)


            for (output, groundtruth, ep_id, step_id) in zip(outputs, groundtruths, ep_ids, step_ids):
                # output = extract_answer(output)
                # print(f"### output: {output}")
                # print(f"### groundtruth: {groundtruth}")
                # input("stop here")
                results.append({
                    "output": output, 
                    "groundtruth": groundtruth, 
                    "ep_id": ep_id, 
                    "step_id": step_id.item()
                })

        return results

    def save(self, path):
        self.accelerator.save_state(path, safe_serialization=False)
        # torch.save(self.agent.critic.state_dict(), os.path.join(path, "critic.pt"))
        # Added: Save policy model in transformers format
        self.agent.model.save_pretrained(path)
        self.agent.policy_processor.save_pretrained(path)


    def load(self, path):
        self.accelerator.load_state(path)
        # critic_path = os.path.join(path, "critic.pt")
        # if os.path.exists(critic_path):
        #     self.agent.critic.load_state_dict(torch.load(critic_path))



def train(agent: Agent, accelerator: Accelerator, config: Dict):
    trainer = QwenPolicyTrainer(
        agent=agent,
        accelerator=accelerator,
        config=config
    )


    trainer.lm_optimizer = accelerator.prepare(trainer.lm_optimizer)

    batch_size = config["batch_size"]
    all_trajectories = utils.read_jsonl(config["data_path"])

    agent.prepare()
    trainer.prepare()

    print(f"### all trajectories: {len(all_trajectories)}")

    logs = []
    train_trajectories = all_trajectories[:int(len(all_trajectories) * 0.95)]
    val_trajectories = all_trajectories[int(len(all_trajectories) * 0.95):]

    random.shuffle(train_trajectories)
    sample_num = config["batch_size"] * config["grad_accum_steps"]

    print(f"batch_size: {batch_size}")
    print(f"grad_accum_steps: {config['grad_accum_steps']}")
    
    for epoch in range(config["epochs"]):
        print(f"### epoch {epoch}")

        print(f"length of train_trajectories: {len(train_trajectories)}")
        print(f"sample_num: {sample_num}")
        print(f"divide: {len(train_trajectories) // sample_num}")
        
        # Training
        print(f"### Training")
        for train_step in tqdm(range(len(train_trajectories) // sample_num), desc=f"Epoch {epoch} Training", disable=not accelerator.is_main_process):
            sample_trajectories = train_trajectories[train_step * sample_num: (train_step + 1) * sample_num]
            # Sample Policy Model outputs
            # Sample Policy Model outputs, adding "output", "groundtruth", "ep_id", "step_id" fields to results
            results = trainer.infer(sample_trajectories, batch_size)

            # print(f"### results: {results}")
            # input("stop here")
            # Add critic_input, policy_output, critic_image to traj data
            sample_trajectories = update_trajectory(sample_trajectories, results)
            
            
            replay_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(sample_trajectories))

            for d in sample_trajectories:
                replay_buffer.insert(**d)
            # Train
            logs.extend(trainer.update_policy(replay_buffer, is_validation=False, batch_size=batch_size))

        # Validation
        print(f"### Validation")
        results = trainer.infer(val_trajectories, batch_size)
        val_trajectories = update_trajectory(val_trajectories, results)
        validation_buffer = ReplayBuffer(batch_size=batch_size, capacity=len(val_trajectories))
        for d in val_trajectories:
            validation_buffer.insert(**d)
        logs.extend(trainer.update_policy(validation_buffer, is_validation=True, batch_size=batch_size))

        if accelerator.is_main_process:
            save_path = config["save_path"]
            print("### saving")
            os.makedirs(save_path, exist_ok=True)
            epoch_path = os.path.join(save_path, f"epoch_{epoch}")
            os.makedirs(epoch_path, exist_ok=True)
            trainer.save(epoch_path)
            utils.write_jsonl(logs, os.path.join(epoch_path, "train_log.jsonl"))
            utils.plot_loss(epoch_path, keys=["train loss", "train Q value", "val loss", "val Q value"])


if __name__ == "__main__":
    # general or webshopping
    config_path = f"configs/train_policy_m2w.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    print(f"### config:")
    for k, v in config.items():
        print(f"\t{k}: {v}")
    
    accelerator = Accelerator(
        # mixed_precision="bf16",
        # split_batches=True,
        # dispatch_for_inference=True,
        gradient_accumulation_steps=config["grad_accum_steps"]
    )
    
    # Modify here, do not specify a specific policy_device, let the model automatically allocate to multiple GPUs
    agent = Agent(
        device=accelerator.device,
        accelerator=accelerator,
        temperature=config["temperature"],
        do_sample=config["do_sample"],
        policy_lm=config["policy_lm"],
        critic_lm=config["critic_lm"],
        max_new_tokens=config["max_new_tokens"],
        policy_device="auto",  # Changed to auto, let the model decide distribution
    )

    train(agent=agent, accelerator=accelerator, config=config)