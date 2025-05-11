import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM
import PIL.Image
from PIL import Image
from qwen_vl_utils import process_vision_info
from typing import List, Union
import os
import torch.nn.functional as F
from accelerate import Accelerator, infer_auto_device_map, dispatch_model



class Agent(torch.nn.Module):
    def __init__(self, device, accelerator, policy_lm, critic_lm, do_sample, temperature, max_new_tokens, policy_device="auto"):
        super(Agent, self).__init__()
        
        print(f"### load policy lm: {policy_lm}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            policy_lm,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        self.policy_processor = AutoProcessor.from_pretrained(
            policy_lm,
            trust_remote_code=True
        )
        
        self.policy_processor.tokenizer.padding_side = 'left'
        
        print(f"### load critic: {critic_lm}")

        self.do_sample = do_sample
        self.temperature = temperature
        self.accelerator = accelerator
        self.softmax = torch.nn.Softmax(dim=-1)
        self.max_new_tokens = max_new_tokens
        
        # vllm api
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.openai_model = critic_lm

    
    def prepare(self):
        self.model = self.accelerator.prepare(self.model)


    def get_log_prob(
        self,
        texts: List[str],
        images: List[Union[str, Image.Image]],
        targets: List[str]
    ):
        log_probs = []

        for text, img, tgt in zip(texts, images, targets):
            # 1) Prepare message dict
            if isinstance(img, str):
                img_path = img if img.startswith("file://") else f"file://{img}"
                img_entry = {"type": "image", "image": img_path}
            else:
                # If PIL.Image, save temporarily or adapt process_vision_info accordingly
                img.save("/tmp/tmp.png")
                img_entry = {"type": "image", "image": "file:///tmp/tmp.png"}

            messages = [{
                "role": "user",
                "content": [img_entry, {"type": "text", "text": text}]
            }]

            prompt = self.policy_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ).strip()

            images_data, videos = process_vision_info(messages) 

            inputs = self.policy_processor(
                text=prompt,
                images=images_data,
                videos=videos,
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(self.model.device)

            tmp = self.model(
                **inputs,
                return_dict=True,
                use_cache=False
            )
            seq_len = tmp.logits.size(1)

            full_labels = torch.full(
                (1, seq_len),
                fill_value=-100,
                device=self.model.device,
                dtype=torch.long
            )

            target_ids = self.policy_processor.tokenizer(
                tgt, add_special_tokens=False
            ).input_ids
            full_labels[0, -len(target_ids):] = torch.tensor(
                target_ids, device=self.model.device  
            )


            outputs = self.model(
                **inputs,
                labels=full_labels,
                return_dict=True,
                use_cache=False
            )
            loss = outputs.loss  

            valid_count = (full_labels != -100).sum().item()
            seq_log_prob = - loss.item() * valid_count
            log_probs.append(seq_log_prob)

        return log_probs
    

    def get_action(self, texts, images):
        """
        batch inference
        """

        processed_images = []
        for img in images:
            if isinstance(img, (PIL.Image.Image, Image.Image)):
                processed_images.append(img)
            else:
                processed_images.append(Image.open(img))
        
        messages = []
        for text, image in zip(texts, processed_images):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text}
                    ]
                }
            ])
        
        batch_texts = [
            self.policy_processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            ) for msg in messages
        ]
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.policy_processor(
            text=batch_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids = inputs.input_ids.to(self.model.device)
        
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature
        )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]

        actions = self.policy_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return actions

    def save(self, path):
        self.model.save_pretrained(path)
        self.policy_processor.save_pretrained(path)

    def load(self, path):
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map={"": self.model.device},
            trust_remote_code=True
        )
        self.policy_processor = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=True
        )
        self.policy_processor.tokenizer.padding_side = 'left'

    def get_critic_response(self, prompt):
        import openai
        
        openai.api_key = self.openai_api_key
        openai.api_base = self.openai_api_base
        
        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that evaluates responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None