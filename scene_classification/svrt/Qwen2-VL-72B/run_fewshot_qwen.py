import numpy as np
import os
import json
import copy
import torch
import math
import time
import pickle
import pandas as pd
import argparse
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils_fewshot_cot0 import get_all_attempts, get_category_from_filename
from qwen_vl_utils import process_vision_info


def construct_convo(incontext_images, query_image):
        instruction_message = '''
        Today, you'll be doing a visual reasoning task. 
        Each problem consists of a series of images. 
        Each image contains multiple objects with relations between them, and belongs to one of two categories (0 or 1).
        The categories are based on the pattern of relations between the objects.
        Your job is to learn to correctly classify the images.
        After viewing a series of images, you will be presented with a target image and asked to classify it.\n
        '''

        context_image_questions = [
            f"Picture-{image_id + 1} belongs to category {get_category_from_filename(filename)}.\n"
            for image_id, filename in enumerate(incontext_images)
        ]
        context_images_message = ''.join(context_image_questions)

        query_message = f'''Picture-{len(incontext_images) + 1} is the target image. Does this image belong to category 0 or 1? 
        Please respond with either "0" or "1" and only one of these two numbers. You must provide a final answer, even if you are uncertain.'''

        message = instruction_message + context_images_message + '\n' + query_message

        # Messages containing multiple images and a text query
        content = []
        for image in incontext_images:
            content.append({"type": "image", "image": image})
        content.append({"type": "image", "image": query_image})
        content.append({"type": "text", "text": message})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        return messages


if __name__ == '__main__':
    seed = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    image_root = "data/svrt"
    save_root = 'results'
    os.makedirs(save_root, exist_ok=True)

    model_name = "Qwen/Qwen2-VL-72B-Instruct-AWQ"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    min_pixels = 224 * 224
    max_pixels = 2048 * 2048
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)

    # All data
    columns = [
        'condition',
        'problem',
        'num_incontext',
        'attempt',

        'incontext_images',
        'query_image',
        'query_category',
        'num_positive',
        'num_negative',
        'convo',

        'selected_category',
        'accuracy',
        'raw_response',
        'elapsed_time'
    ]
    data = pd.DataFrame(columns=columns)

    total_problems = 23
    total_incontexts = 9
    total_attempts = 10

    cotN = 'chain_of_thought0'
    range_problems = range(1, total_problems + 1)
    range_incontexts = range(1, total_incontexts + 1)
    range_attempts = range(1, total_attempts + 1)
    os.makedirs(os.path.join(save_root, cotN), exist_ok=True)

    #########################################################################################################
    # Run
    #########################################################################################################
    for problem in range_problems:

        # Generate images for all attempts
        dict_all_attempts = get_all_attempts(
            image_root,
            problem,
            total_incontexts,
            total_attempts
        )

        # Save structure of all attempts
        with open(f'{save_root}/{cotN}/{problem}_dict_all_attempts.pkl', 'wb') as file:
            pickle.dump(dict_all_attempts, file)

        #########################################################################################################
        # Iterate over all 10 number of in-contexts and attempts
        #########################################################################################################
        for num_incontext in range_incontexts:
            for attempt in range_attempts:
                print(f'----------------------- Problem: {problem}; Number of in-context: {num_incontext}; Attempt: {attempt} ---------------------------')

                start_time = time.time()

                incontext_and_query_images = dict_all_attempts[num_incontext][attempt]

                # In-context image filenames are the first num_incontext, the query image filename is num_incontext+1 (last)
                incontext_images = incontext_and_query_images[:num_incontext]
                query_image = incontext_and_query_images[-1]

                # Query category ground-truth
                incontext_categories = [get_category_from_filename(incontext_image) for incontext_image in
                                        incontext_images]
                query_category = get_category_from_filename(query_image)

                # How many pos and neg in incontext images
                num_positive = np.sum(incontext_categories)
                num_negative = len(incontext_images) - np.sum(incontext_categories)

                # Get convo
                messages = construct_convo(incontext_images, query_image)

                # Preparation for inference
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                # Ensure that trial response is a single number, 0 or 1
                response_inappropriate = True
                num_trials = 1
                while response_inappropriate:
                    try:
                        # Inference
                        generated_ids = model.generate(**inputs, max_new_tokens=512)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        response = processor.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]

                        # Parse accuracy -- this line assumes it is integer, so will raise error if not
                        selected_category = int(response)

                        # Check if accuracy is either 0 or 1 -- this line will raise error if not
                        assert selected_category in [0, 1]

                        # Determine accuracy
                        accuracy = 1 if selected_category == query_category else 0

                        # If code gets to this point, then while-loop condition can be set to false; no need to re-run API
                        response_inappropriate = False

                    except Exception as e:
                        print(f'Re-running trial {num_trials}. Response is {response}')
                        num_trials += 1
                        continue

                # Print everything
                print('In-context images:', incontext_images)
                print('Query image:', query_image)
                print('Number of positive:', num_positive)
                print('Number of negative:', num_negative)
                print('In-context categories:', incontext_categories)
                print('Query category:', query_category)
                print(f'Assistant: {response}')
                print('Accuracy:', accuracy)

                #########################################################################################################
                # Save information
                #########################################################################################################
                elapsed_time = time.time() - start_time
                print(f'Elapsed time: {elapsed_time:.02f}')

                # Add row for attempt information
                row = [
                    cotN,
                    problem,
                    num_incontext,
                    attempt,

                    incontext_images,
                    query_image,
                    query_category,
                    num_positive,
                    num_negative,
                    messages,

                    selected_category,
                    accuracy,
                    response,
                    elapsed_time
                ]
                data.loc[len(data.index) + 1] = row

        # Re-save the full dataframe at each attempt
        save_path = f'{save_root}/{cotN}/{problem}_fewshot_results.csv'
        data.to_csv(save_path)
