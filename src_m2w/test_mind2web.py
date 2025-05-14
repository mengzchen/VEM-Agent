# evaluation on mind2web
import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
from PIL import Image
import numpy as np
from prompt import action_prompt
from qwen_vl_utils import process_vision_info
from utils import extract_answer

logging.basicConfig(level=logging.INFO)


def action2step(action, image_size, return_bbox=False):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
                action["bbox"]["y"] + action["bbox"]["height"]]
        bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
        bbox = [round(item, 3) for item in bbox]

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point,
                                                                                               select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point,
                                                                                               typed_text)

    if return_bbox:
        return action_step, bbox
    else:
        return action_step


def action2step2(action, image_size):
    action_type = action["action_type"]
    assert action_type in ['click', 'type', 'select']

    point_x, point_y = action["click_point"]
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])


    if action_type == 'click':
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'select':
        select_value = action["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point,
                                                                                               select_value)
    elif action_type == 'type':
        typed_text = action["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point,
                                                                                               typed_text)
    return action_step

def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True)
args = parser.parse_args()

model_path = " /model/path"

mind2web_imgs_dir = "../mm-mind2web/mind2web_images"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, device_map="cuda", torch_dtype="auto").eval()
processor = AutoProcessor.from_pretrained(model_path)

mind2web_test = json.load(open('../mm-mind2web/mind2web_data_test_' + args.task + '.json', 'r'))


results = []
for episode in tqdm(mind2web_test):
    goal = episode["confirmed_task"]
    annot_id = episode["annotation_id"]
    previous_actions = []
    results_actions = []

    for j, step in enumerate(episode["actions"]):
        if "bbox" not in step:
            print("action not found")
            continue

        filename = annot_id + '-' + step["action_uid"] + '.jpg'
        img_path = os.path.join(mind2web_imgs_dir, filename)
        if not os.path.exists(img_path):
            print("img not found")
            continue
        image = Image.open(img_path)

        previous_step = ""
        for i, action in enumerate(previous_actions[-4:]):
            previous_step += 'Step' + str(i) + ': ' + action + ". "

        action_step = action2step(step, image.size)
        previous_actions.append(action_step)

        prompt = action_prompt.replace("{task}", goal).replace("{action_history}", previous_step)

        action_step_ref, bbox_ref = action2step(step, image.size, return_bbox=True)
        try:
            action_step_ref = ast.literal_eval(action_step_ref)
        except:
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # print(f"### response: {response}")

        step_result = {"annot_id": annot_id, "img_path": img_path, "instruction": goal, "sentence": response,
                       "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"]]}
        try:
            action_pred = extract_answer(response)
            action_pred = json.loads(action_pred)
            image = Image.open(img_path)
            action_pred = action2step2(action_pred, image.size)
            action_pred = ast.literal_eval(action_pred)


            if action_pred["action_type"] == action_step_ref["action_type"]:
                step_result["Op_match"] = True

            click_point = action_pred["click_point"]

            if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                step_result["Ele_match"] = True

            pred_str = str(action_pred["action_type"])
            if action_pred["action_type"] == 3 or action_pred["action_type"] == 2:
                pred_str += ' '
                pred_str += action_pred["value"].lower()
            ref_str = str(action_step_ref["action_type"])
            if action_step_ref["action_type"] == 3 or action_step_ref["action_type"] == 2:
                ref_str += ' '
                ref_str += action_step_ref["value"].lower()

            op_f1 = calculate_f1(pred_str, ref_str)
            step_result["Op_F1"][0] = op_f1

        except Exception as e:
            logging.info(f"format wrong: {e}")

        logging.info(step_result)

        results_actions.append(step_result)

    results.append(results_actions)


# calculate metrics
num_step = 0
num_episode = 0
num_op = 0
num_ele = 0
op_f1 = {4: [], 2: [], 3: []}
macro_ele_acc = {}
macro_step_acc = {}
macro_action_f1 = {}
num_step_success = 0
num_episode_success = 0
for i, item in enumerate(results):
    macro_ele_acc[i] = []
    macro_step_acc[i] = []
    macro_action_f1[i] = []
    num_episode += 1
    episode_success = True
    for step_result in item:
        num_step += 1

        if step_result["Op_match"]:
            num_op += 1

        if step_result["Ele_match"]:
            num_ele += 1
            macro_ele_acc[i].append(1)
        else:
            macro_ele_acc[i].append(0)

        if step_result["Op_F1"][1] in op_f1:
            op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
        macro_action_f1[i].append(step_result["Op_F1"][0])

        if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
            num_step_success += 1
            macro_step_acc[i].append(1)
        else:
            macro_step_acc[i].append(0)
            episode_success = False

    if episode_success:
        num_episode_success += 1

marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])

logging.info("Operation F1: " + str(marco_op_f1))
logging.info("Element Acc: " + str(num_ele / num_step))
logging.info("Step Success: " + str(num_step_success / num_step))
logging.info("Episode Success: " + str(num_episode_success / num_episode))
logging.info("Operation F1 cate: " + str([np.mean(x) for x in op_f1.values()]))

macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])
logging.info("Macro Ele Acc: " + str(macro_ele_acc))
logging.info("Macro Op F1: " + str(macro_action_f1))
logging.info("Macro Step SR: " + str(macro_step_acc))