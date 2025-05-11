import os
import sys
sys.path.insert(0, os.getcwd())

from tqdm import tqdm
import json
import threading
import random
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import utils

from prompt import (
    prompt_critic_system, prompt_critic_user
)
from gpt_scorer import GPTScorer


def get_unfinish_anns(anns, rpath):
    if os.path.exists(rpath):
        unfinish_anns = []
        finish_anns = utils.read_jsonl(rpath)
        finish_ids = [f"{ann['ep_id']}_{ann['step_id']}" for ann in finish_anns]
        for ann in anns:
            if f"{ann['ep_id']}_{ann['step_id']}" in finish_ids:
                pass
            else:
                unfinish_anns.append(ann)
        print(f"### Finished annotations: {len(finish_anns)} Unfinished: {len(unfinish_anns)}")
        return unfinish_anns
    else:
        return anns


# convert action to prediction format
def action2step(action, image_size):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']  # five types of data

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x , point_y]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "[{},{}]".format(click_point[0], click_point[1])

    if action_type in ['CLICK', 'HOVER', 'ENTER']:  # following mind2web, these three actions are regarded as click
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"\"}}".format('"click"', click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format('"select"', click_point, select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format('"type"', click_point, typed_text)
    return action_step


class Mind2Web:
    def __init__(self, split: str):
        self.image_dir = "../mm-mind2web/mind2web_images"
        self.split = split
        
        os.makedirs("../mm-mind2web/mind2web_anns", exist_ok=True)
        
        self.gpt = GPTScorer()
        

    def get_unfold_data(self):
        print(f"### Starting to unfold {self.split} dataset...")
        anns = utils.read_json(f"../mm-mind2web/mind2web_data_{self.split}.json")
        steps = []
        
        print(f"### Processing {len(anns)} episodes...")
        for episode in tqdm(anns, desc="Processing episodes"):

            action_list, image_list, add_point_image_list = [], [], []
            action_desc_list = []

            goal = episode["confirmed_task"]
            annot_id = episode["annotation_id"]

            # New: Check if all steps are valid, otherwise discard the entire episode
            episode_valid = True
            for step in episode["actions"]:
                if "bbox" not in step:
                    episode_valid = False
                    break
                img_filename = f"{annot_id}-{step['action_uid']}.jpg"
                image_path = os.path.join(self.image_dir, img_filename).replace("\\", "/")
                if not os.path.exists(image_path):
                    episode_valid = False
                    break
            if not episode_valid:
                continue  # Discard the entire episode

            for step_id, step in enumerate(episode["actions"]):
                img_filename = f"{annot_id}-{step['action_uid']}.jpg"
                image_path = os.path.join(self.image_dir, img_filename).replace("\\", "/")
                image = Image.open(image_path)
                action_dict = {
                    "operation": step["operation"],
                    "bbox": step["bbox"]
                }
                action_list.append(action_dict)
                image_list.append(image_path)
                action_desc = action2step(step, image.size)
                action_desc_list.append(f"{action_desc}")
                add_point_image_list.append(utils.add_visilize2screenshot(image_path, step, "score"))
            
                steps.append({
                    "ep_id": annot_id,
                    "step_id": step_id,
                    "task": goal,
                    "action_list": action_list,
                    "action_desc_list": action_desc_list,
                    "image_list": image_list,
                    "add_point_image_list": add_point_image_list
                })


        utils.write_jsonl(steps, f"../mm-mind2web/mind2web_anns/{self.split}.jsonl")
        print(f"### Completed unfolding {self.split} dataset, generated {len(steps)} records")
        return steps  # Add this line to return processed data

    def get_gpt_label(self):
        print(f"### Starting to get GPT scores for {self.split} dataset...")
        anns = utils.read_jsonl(f"../mm-mind2web/mind2web_anns/{self.split}.jsonl")
        ann_wpath = f"../mm-mind2web/mind2web_anns/{self.split}_critic_positive.jsonl"
        unfinish_anns = get_unfinish_anns(anns, ann_wpath)

        write_lock = threading.Lock()

        def process_ann(ann):
            """
            Process a single record
            """
            # response = self.gpt.get_score_m2w(ann)
            # response = utils.parse_response(response)
            # ann["critic_output"], ann["critic_explanation"] = response["rating"], response["explanation"]

            ann["critic_output"], ann["critic_explanation"] = 2, "GPT score: 2"

            history = ""
            for i, action in enumerate(ann["action_desc_list"][:ann["step_id"]]):
                history += f"step {i}: {action}\n"

            conversations = [
                {"from": "human", "value": prompt_critic_system + prompt_critic_user.format(ann["task"], history, ann["action_desc_list"][ann["step_id"]])},
                {"from": "gpt", "value": "2"}
            ]
            ann["critic_inputs"] = conversations
            ann["critic_images"] = ann["add_point_image_list"][ann["step_id"]].replace("\\", "/")

            return ann
       

        with open(ann_wpath, "a") as fout:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ann = {executor.submit(process_ann, ann): ann for ann in unfinish_anns}
                for future in tqdm(as_completed(future_to_ann), total=len(future_to_ann)):
                    ann = future_to_ann[future]
                    try:
                        result = future.result()
                        with write_lock:
                            fout.writelines(json.dumps(result) + "\n")
                    except Exception as exc:
                        # pass
                        print(f'Error processing annotation {ann}: {exc}')
            fout.close()
        print(f"### Completed GPT scoring, results saved to {ann_wpath}")

    def get_rl_data(self):
        print(f"### Starting to generate RL training data for {self.split} dataset...")
        ann_rpath = f"../mm-mind2web/mind2web_anns/{self.split}.jsonl"
        ann_wpath = f"../mm-mind2web/mind2web_anns/{self.split}_policy.jsonl"
        anns = utils.read_jsonl(ann_rpath)
        print(f"ann keys: {anns[0].keys()}")

        print(f"### Converting {len(anns)} records...")
        valid_anns = []  # Collect valid annotations
        
        for ann in tqdm(anns, desc="Generating RL data"):
            try:
                # Unified data validity check
                if not isinstance(ann.get("step_id", None), int):
                    print(f"Invalid step_id: {ann.get('step_id')} in ann {ann.get('ep_id')}")
                    continue
                    
                if "action_desc_list" not in ann or "image_list" not in ann:
                    print(f"Missing required keys in ann {ann.get('ep_id')}")
                    continue
                    
                if ann["step_id"] >= len(ann["action_desc_list"]) or ann["step_id"] >= len(ann["image_list"]):
                    print(f"Step_id out of range: {ann.get('ep_id')}_{ann.get('step_id')}")
                    continue
                
                # Construct previous actions
                previous_actions = []
                for i in range(ann["step_id"]):
                    if i >= len(ann["action_desc_list"]):
                        raise IndexError(f"Previous action index out of range: {i}")
                    previous_actions.append(ann['action_desc_list'][i])

                previous_actions_str = ""

                for i, action in enumerate(previous_actions):
                    previous_actions_str += f"step {i}: {action}\n"
                
                # Construct policy_input
                ann["policy_input"] = {
                    "task": ann["task"],
                    "previous_actions": previous_actions_str
                }
                
                # Construct policy_output
                current_action = ann["action_desc_list"][ann["step_id"]]

                ann["policy_output"] = current_action
                
                # policy_image uses the current step's image
                ann["policy_image"] = ann["image_list"][ann["step_id"]]
                
                valid_anns.append(ann)
                
            except Exception as e:
                print(f"Error processing ann {ann.get('ep_id')}_{ann.get('step_id')}: {e}")
                continue
        
        utils.write_jsonl(valid_anns, ann_wpath)
        print(f"### Completed RL data generation, valid records: {len(valid_anns)}/{len(anns)}, results saved to {ann_wpath}")

    def get_negative_anns(self, num):
        """
        Generate negative samples, following the get_gpt_label pattern, adapted for mind2web paths and fields
        """
        ann_rpath = f"../mm-mind2web/mind2web_anns/{self.split}_critic.jsonl"
        ann_wpath = f"../mm-mind2web/mind2web_anns/{self.split}_critic_negative.jsonl"

        step_ids = []
        anns = utils.read_jsonl(ann_rpath)
        for ann in anns:
            if ann.get("critic_output", None) == 2:
                step_ids.append(f"{ann['ep_id']}_{ann['step_id']}")
        step_ids = step_ids[:num]

        anns = [ann for ann in anns if f"{ann['ep_id']}_{ann['step_id']}" in step_ids]
        unfinish_anns = get_unfinish_anns(anns, ann_wpath)

        write_lock = threading.Lock()

        def process_ann(ann):
            # Get negative sample actions and images
            negative_action, negative_add_point_image_path = self.gpt.get_negative_action_m2w(ann)
            new_prompt = prompt_critic_system + prompt_critic_user.format(
                ann["task"], "\n".join(ann["action_desc_list"][:ann["step_id"]]), negative_action
            )
            conversations = [
                {"from": "human", "value": new_prompt},
                {"from": "gpt", "value": "1"}
            ]
            ann["critic_inputs"] = conversations
            ann["critic_images"] = negative_add_point_image_path.replace("\\", "/")
            # Mark as negative sample
            ann["critic_output"] = 1
            ann["critic_explanation"] = "GPT score: 1"
            return ann

        with open(ann_wpath, "a") as fout:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ann = {executor.submit(process_ann, ann): ann for ann in unfinish_anns}
                for future in tqdm(as_completed(future_to_ann), total=len(future_to_ann)):
                    ann = future_to_ann[future]
                    try:
                        result = future.result()
                        with write_lock:
                            fout.writelines(json.dumps(result) + "\n")
                    except Exception as exc:
                        print(f'Error processing annotation {ann}: {exc}')
            fout.close()

    def get_negative_label(self):
        """
        Generate negative samples by modifying original actions to create unreasonable operations
        """
        print(f"### Starting to generate negative samples for {self.split} dataset...")
        ann_rpath = f"../mm-mind2web/mind2web_anns/{self.split}.jsonl"
        ann_wpath = f"../mm-mind2web/mind2web_anns/{self.split}_critic_negative.jsonl"

        anns = utils.read_jsonl(ann_rpath)
        # Randomly select a portion of samples as the basis for negative samples
        sample_num = min(len(anns), 5000)  # Limit maximum sample size
        sampled_anns = random.sample(anns, sample_num)
        unfinish_anns = get_unfinish_anns(sampled_anns, ann_wpath)

        write_lock = threading.Lock()

        def process_ann(ann):
            """
            Process a single record to generate a negative sample
            """
            try:
                # Get a copy of the current action for modification
                current_action_str = ann["action_desc_list"][ann["step_id"]]
                current_action = json.loads(current_action_str)
                
                # Create negative sample action
                negative_action = create_negative_action(current_action)
                negative_action_str = json.dumps(negative_action)
                
                # Build history
                history = ""
                for i, action in enumerate(ann["action_desc_list"][:ann["step_id"]]):
                    history += f"step {i}: {action}\n"
                
                # Create negative sample conversation input
                conversations = [
                    {"from": "human", "value": prompt_critic_system + prompt_critic_user.format(
                        ann["task"], history, negative_action_str)},
                    {"from": "gpt", "value": "1"}  # Negative sample, score is 1
                ]
                
                # Copy the current annotation for modification, avoid modifying original data
                negative_ann = ann.copy()
                negative_ann["critic_inputs"] = conversations
                
                # Generate new annotated image based on modified coordinates
                current_image_path = ann["image_list"][ann["step_id"]]
                
                # Create a mock action object for generating new annotated image
                mock_action = {
                    "operation": {"original_op": current_action.get("action_type", "CLICK").upper()},
                    "bbox": {
                        "x": float(negative_action["click_point"][0]) - 5,  # Simple estimation of bbox size
                        "y": float(negative_action["click_point"][1]) - 5,
                        "width": 10,
                        "height": 10
                    }
                }
                
                # Use utils.add_visilize2screenshot to generate new annotated image
                negative_image_path = utils.add_visilize2screenshot(current_image_path, mock_action, "negative")
                negative_ann["critic_images"] = negative_image_path.replace("\\", "/")
                
                negative_ann["critic_output"] = 1  # Mark as negative sample
                negative_ann["critic_explanation"] = f"GPT score: 1"
                negative_ann["original_action"] = current_action_str
                negative_ann["negative_action"] = negative_action_str
                
                return negative_ann
            except Exception as e:
                print(f"Error while processing sample {ann.get('ep_id')}_{ann.get('step_id')}: {e}")
                return None

        def create_negative_action(action):
            """
            Create negative action based on original action
            Strategies:
            1. Randomly change action type
            2. Randomly move click position
            3. If type or select, possibly change input value
            """
            negative_action = action.copy()
            strategy = random.choice(["change_type", "change_point", "change_value"])
            from faker import Faker
            fake = Faker()
            
            if strategy == "change_type":
                # Change action type
                original_type = action["action_type"]
                action_types = ["click", "type", "select"]
                action_types.remove(original_type)
                negative_action["action_type"] = random.choice(action_types)
                
                # Ensure type and select actions have values
                if negative_action["action_type"] == "type" and not negative_action.get("value"):
                    negative_action["value"] = fake.word()
                elif negative_action["action_type"] == "select" and not negative_action.get("value"):
                    negative_action["value"] = fake.word()
            
            elif strategy == "change_point":
                # Change click position, move far away from original position
                try:
                    original_point = action["click_point"]
                    if isinstance(original_point, str):
                        original_point = json.loads(original_point)
                    
                    # Ensure click point is valid
                    if isinstance(original_point, list) and len(original_point) == 2:
                        # Randomly move click position by 100-500 pixels
                        x_offset = random.randint(100, 500) * random.choice([-1, 1])
                        y_offset = random.randint(100, 500) * random.choice([-1, 1])
                        
                        # Ensure new position is within reasonable range (assuming webpage width and height are 1920x1080)
                        new_x = max(0, min(1280, float(original_point[0]) + x_offset))
                        new_y = max(0, min(720, float(original_point[1]) + y_offset))
                        
                        negative_action["click_point"] = [new_x, new_y]
                except:
                    # Handle click point parsing failure
                    negative_action["click_point"] = [random.randint(50, 1000), random.randint(50, 700)]
            
            elif strategy == "change_value" and action.get("value"):
                # Change input value or selection value
                original_value = action["value"]
                
                if action["action_type"] == "type":
                    negative_action["value"] = fake.word()
                
                elif action["action_type"] == "select":
                    # For dropdowns, generate a possibly non-existent option
                    negative_action["value"] = fake.word()
        
            return negative_action


        with open(ann_wpath, "a") as fout:
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ann = {executor.submit(process_ann, ann): ann for ann in unfinish_anns}
                for future in tqdm(as_completed(future_to_ann), total=len(future_to_ann)):
                    ann = future_to_ann[future]
                    try:
                        result = future.result()
                        if result:  # Ensure processing was successful
                            with write_lock:
                                fout.writelines(json.dumps(result) + "\n")
                    except Exception as exc:
                        print(f'Error processing sample {ann}: {exc}')
            fout.close()
        
        print(f"### Completed negative sample generation, results saved to {ann_wpath}")


if __name__ == "__main__":

    train_data = Mind2Web(split="train")
    # train_data.get_unfold_data()
    # train_data.get_gpt_label()
    # train_data.get_negative_label()
    # train_data.get_gpt_label()
    # train_data.get_negative_anns(400)
    # train_data.get_negative_anns(2000)
    train_data.get_rl_data()

