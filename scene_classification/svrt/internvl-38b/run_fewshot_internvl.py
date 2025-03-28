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
from transformers import AutoModel, AutoTokenizer
from utils_fewshot_cot0 import get_all_attempts, get_category_from_filename

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')

    # TODO: remove this line for other images
    image = image.resize((input_size, input_size))

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def read_images(images, input_size=448, max_num=12):
    pixel_values = []
    num_patches_list = []
    for image in images:
        pixel_val = load_image(image, input_size=input_size, max_num=max_num).to(torch.bfloat16).cuda()
        pixel_values.append(pixel_val)
        num_patches_list.append(pixel_val.size(0))

    pixel_values = torch.cat(pixel_values, dim=0)
    return pixel_values, num_patches_list


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def construct_convo(incontext_images):
        instruction_message = '''
        Today, you'll be doing a visual reasoning task. 
        Each problem consists of a series of images. 
        Each image contains multiple objects with relations between them, and belongs to one of two categories (0 or 1).
        The categories are based on the pattern of relations between the objects.
        Your job is to learn to correctly classify the images.
        After viewing a series of images, you will be presented with a target image and asked to classify it.\n
        '''

        context_image_questions = [
            f"Image-{image_id + 1}: <image> This image belongs to category {get_category_from_filename(filename)}.\n"
            for image_id, filename in enumerate(incontext_images)
        ]
        context_images_message = ''.join(context_image_questions)

        query_message = f'''Image-{len(incontext_images) + 1}: <image> Here is the target image. Does this image belong to category 0 or 1? 
        Please respond with either "0" or "1" and only one of these two numbers. You must provide a final answer, even if you are uncertain.'''

        message = instruction_message + context_images_message + '\n' + query_message

        return message


if __name__ == '__main__':
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Inference InternVL 2.5")

    # Add arguments
    parser.add_argument('--model', type=str, default='38B', choices=['8B', '38B'], help="Which InternVL 2.5 model to use")

    # Parse the arguments
    args = parser.parse_args()

    seed = 3
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    image_root = "data/svrt"
    save_root = 'results'
    os.makedirs(save_root, exist_ok=True)

    if args.model == '38B':
        path = 'OpenGVLab/InternVL2_5-38B-MPO'
        device_map = split_model('InternVL2_5-38B')
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map
        ).eval()
    elif args.model == '8B':
        path = 'OpenGVLab/InternVL2_5-8B'
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).cuda().eval()
    else:
        raise NotImplementedError(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

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
                message = construct_convo(incontext_images)

                pixel_values, num_patches_list = read_images(incontext_and_query_images)
                print(pixel_values.shape)

                # Ensure that trial response is a single number, 0 or 1
                response_inappropriate = True
                num_trials = 1
                while response_inappropriate:
                    try:
                        generation_config = dict(max_new_tokens=1024, do_sample=True)
                        response, history = model.chat(tokenizer, pixel_values, message, generation_config,
                                                       num_patches_list=num_patches_list,
                                                       history=None, return_history=True)

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
                    message,

                    selected_category,
                    accuracy,
                    response,
                    elapsed_time
                ]
                data.loc[len(data.index) + 1] = row

        # Re-save the full dataframe at each attempt
        save_path = f'{save_root}/{cotN}/{problem}_fewshot_results.csv'
        data.to_csv(save_path)
