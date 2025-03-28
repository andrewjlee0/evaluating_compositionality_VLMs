import openai
from openai import AzureOpenAI
import base64
from mimetypes import guess_type
import glob
import pandas as pd
import os 
import re
import numpy as np
import json
import sys
import pickle
import time
import random
import copy
from collections import defaultdict

from utils_fewshot_cot1 import *


#########################################################################################################
# API Setup
#########################################################################################################
api_base = '' # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
api_key = ''
deployment_name = ''
api_version = '2024-02-01' # this might change in the future

client = AzureOpenAI(
    api_key = api_key,  
    api_version = api_version,
    base_url = f'{api_base}/openai/deployments/{deployment_name}'  # for some reason, changing this to base_url fixes a problem where GPT does not want to describe images with humans
)


#########################################################################################################
# Setup variables
#########################################################################################################
instruction_message = '''
Today, you'll be doing a visual reasoning task. 
Each problem consists of a series of images. 
Each image contains multiple objects with relations between them, and belongs to one of two categories (0 or 1).
The categories are based on the pattern of relations between the objects.
Your job is to learn to correctly classify the images.
After viewing a series of images, you will be presented with a target image and asked to classify it.
When viewing the series of images, you must provide a description of each image.
Make sure to be as detailed as possible.
'''

incontext_message = '''Here is the {idx_incontext_ordinal} image. This image belongs to category {category}. You must provide a description of the image, and are capable of doing so. Make sure to be as detailed as possible. You cannot say that you are unable to provide a description.'''

query_message = '''Here is the target image. Does this image belong to category 0 or 1? Please respond with either "0" or "1" and only one of these two numbers. You must provide a final answer, even if you are uncertain.'''

convo = [
    {"role": "system", "content": instruction_message}
]

#########################################################################################################
# Setup
#########################################################################################################
np.random.seed(3)

# All data
columns = [
    'cotN',
    'test_condition',
    'idx_test_item',

    'concept',
    'relation',
    'object',
    
    'incontext_categories',
    'num_positive',
    'num_negative',
    'query_category',
    'convo',

    'selected_category',
    'accuracy',
    'raw_response',
    'bad_request_error',
    'elapsed_time'
]
data = pd.DataFrame(columns=columns)

# conditions = ['chain_of_thought0', 'chain_of_thought1', 'chain_of_thought2', 'chain_of_thought3']
cotN = 'chain_of_thought1'


#########################################################################################################
# Bongard-HOI
#########################################################################################################
import yaml
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.Bongard_HOI.datasets.image_bongard_bbox import *
from data.Bongard_HOI.datasets.datasets import make

# Get dataset configurations
config_file_path = '../bongard_hoi/data/Bongard_HOI/configs/train_cnn_imagebd.yaml'

# Open the file and load the configuration
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

# All conditions in test
test_type_list = [
    'test_seen_obj_seen_act',
    # 'test_seen_obj_unseen_act',
    # 'test_unseen_obj_seen_act',
    # 'test_unseen_obj_unseen_act'
]

# Create dataset objects and put them into dictionary
test_dataset_dict = {}
for test_condition in test_type_list:

    # Edit a few configurations from YAML file
    dataset_configs = config['{}_dataset_args'.format(test_condition)]

    new_split_file_path = '../bongard_hoi/data/Bongard_HOI' + dataset_configs['im_dir'][1:] + '/hake'
    dataset_configs['im_dir'] = new_split_file_path

    new_split_file_path = '../bongard_hoi/data/Bongard_HOI' + dataset_configs['split_file'][1:]
    dataset_configs['split_file'] = new_split_file_path

    # Construct dataset and append to dict
    test_dataset = make(config['{}_dataset'.format(test_condition)], **dataset_configs)
    test_dataset_dict[test_condition] = test_dataset


#########################################################################################################
# Sample 10 per concept
#########################################################################################################
unique_test_concepts = defaultdict(int)
# with open(f'results/{cotN}/concept_count.pkl', 'rb') as file:
#     unique_test_concepts = pickle.load(file)
# unique_test_concepts = defaultdict(int, unique_test_concepts)


#########################################################################################################
# Run
#########################################################################################################
for test_condition, test_dataset in test_dataset_dict.items():

    # Reset data to be not inclusive of previous conditions
    data = pd.DataFrame(columns=columns)

    for idx_test_item, test_item in enumerate(test_dataset):
        print(idx_test_item)

        # if test_condition == 'test_seen_obj_unseen_act' and idx_test_item < 188:
        #     continue
        # elif test_condition == 'test_unseen_obj_unseen_act' and idx_test_item == 188:
        #     data = pd.read_csv(f'results/{cotN}/test_seen_obj_unseen_act_fewshot_results.csv')
        #     data = data.drop(columns = ['Unnamed: 0'])
        #     print(data.columns)

        concept = test_item['concept']
        relation = test_item['relation']
        object_ = test_item['object']

        # Do all concepts 10 times each
        if unique_test_concepts[concept] >= 10:
            print(f"Concept '{concept}' has been seen 10 times.")
            print()
            continue
        print(f'Count for concept {concept}:', unique_test_concepts[concept])


        print('Chain of Thought Condition:', cotN)
        print('Test item:', f'{idx_test_item + 1} / {len(test_dataset)}')
        print('Test condition:', test_condition)
        print('Concept:', concept)
        print('Relation:', relation)
        print('Object:', object_)

        # Equivalent to MATLAB's tic
        start_time = time.time()

        # In context images
        pos_incontext_images = test_item['pos_shot_ims'][:5]  # 6 total, but get 5
        neg_incontext_images = test_item['neg_shot_ims'][:5]  # 6 total, but get 5

        # Include category for each image
        pos_incontext_images_and_labels = [(1, image_tensor) for image_tensor in pos_incontext_images]
        neg_incontext_images_and_labels = [(0, image_tensor) for image_tensor in neg_incontext_images]

        # Randomly choose one of the lists and remove the last element from the chosen list, to make room for query image 
        chosen_list = np.random.choice(['pos_incontext_images_and_labels', 'neg_incontext_images_and_labels'])

        # Combine the modified chosen list with the other list
        if chosen_list == 'pos_incontext_images_and_labels':
            pos_incontext_images_and_labels = pos_incontext_images_and_labels[:-1]
        else:
            neg_incontext_images_and_labels = neg_incontext_images_and_labels[:-1]
        incontext_images_and_labels = pos_incontext_images_and_labels + neg_incontext_images_and_labels

        # Shuffle the list
        np.random.shuffle(incontext_images_and_labels)

        # Get categories of incontext images, in order
        incontext_categories = [category for category, image in incontext_images_and_labels]
        pos_categories = [category for category, image in pos_incontext_images_and_labels]
        neg_categories = [category for category, image in neg_incontext_images_and_labels]

        # How many positive and negative incontext?
        num_positive = np.sum(incontext_categories)
        num_negative = len(incontext_categories) - np.sum(incontext_categories)
        
        # Print everything
        print('In-context categories:', incontext_categories)
        print('Positive categories:', pos_categories)
        print('Negative categories:', neg_categories)
        print('Number of positive:', num_positive)
        print('Number of negative:', num_negative)


        #########################################################################################################
        # In-Context Interactions
        #########################################################################################################
        bad_request_error = False
        
        # Incontext images + GPT's descriptions of each, iteratively
        for idx_incontext, (incontext_image_category, incontext_image) in enumerate(incontext_images_and_labels):

            if bad_request_error:
                break 

            # Convo with incontext image and message 
            convo_IC = update_convo(
                convo, 
                incontext_image = incontext_image,
                incontext_image_category = incontext_image_category,
                incontext_message = incontext_message,
                idx_incontext = idx_incontext + 1,
            )

            num_calls_to_API = 1
            while True:

                # Keep calling API until it works or GPT's response format is appropriate
                try:   

                    # GPT-4's raw response to incontext image
                    temperature = 0
                    raw_response_IC = get_gpt4v_response(
                        convo_IC,
                        client = client,
                        deployment_name = deployment_name,
                        temperature = temperature
                    )

                    # Description of incontext image
                    assistant_message = raw_response_IC['choices'][0]['message']['content']
                    print(f'GPT Description of In-Context Image {idx_incontext + 1}:', assistant_message)
                    if "sorry" in assistant_message:
                        raise Exception(assistant_message)

                    # If code gets to this point, then while-loop condition can be set to false; no need to re-run API
                    break

                except openai.BadRequestError as bad_request_error_message:
                    print(bad_request_error_message)

                    selected_category = None
                    accuracy = None
                    raw_response_Q = bad_request_error_message
                    bad_request_error = True

                    break

                # KeyboardInterrupt does not trigger this except so you can kill python file when desired
                except Exception as e:
                    print(e)

                    # Exponential backoff with jitter; don't hammer API with calls
                    time.sleep(1 * 2 ** num_calls_to_API + random.uniform(0, 1))
                    print(f'Re-running API call. This is call {num_calls_to_API}.')
                    num_calls_to_API += 1

                    continue

            # Convo with incontext image and message, and GPT's description
            convo_ICR = update_convo(
                convo_IC,
                assistant_message = assistant_message
            )
            # print('Convo with incontext message and GPTs description:', convo_ICR)

            # Save all conversation
            convo = copy.deepcopy(convo_ICR)

        #########################################################################################################
        # If policy warning, then save and move onto next test item
        #########################################################################################################
        if bad_request_error:

            for query_category in [1, 0]:    

                # How many API calls for that trial?
                print('Total number of API calls for this trial:', num_calls_to_API)
                
                elapsed_time = time.time() - start_time
                print('Elapsed time:', elapsed_time)

                # Add row for attempt information
                row = [
                    cotN,
                    test_condition,
                    idx_test_item,

                    concept,
                    relation,
                    object_,
                    
                    incontext_categories,
                    num_positive,
                    num_negative,
                    query_category,
                    convo,

                    selected_category,
                    accuracy,
                    raw_response_Q,
                    bad_request_error,
                    elapsed_time
                ]
                data.loc[len(data.index) + 1] = row

                # Re-save the full dataframe at each attempt
                save_path = f'results/{cotN}/{test_condition}_fewshot_results.csv'
                data.to_csv(save_path)

            # Reset convo
            convo = [
                {"role": "system", "content": instruction_message}
            ]

            print()
            continue


        #########################################################################################################
        # Query interactions
        #########################################################################################################
        # Iterate over query images
        for i, query_category in enumerate([1, 0]):

            print('Query image category:', query_category)

            # Get query image
            query_image = test_item['query_ims'][i]  # first element is always pos, second is always neg

            # Convo with query image and message
            convo = copy.deepcopy(convo_ICR)
            convo_Q = update_convo(
                convo,
                query_image = query_image,
                query_message = query_message
            )
            # print('Convo with query:', len(convo_Q))

            # Query image
            bad_request_error = False
            while True:

                # Keep calling API until it works or GPT's response format is appropriate
                try:    

                    # GPT's raw response to query image and message
                    raw_response_Q = get_gpt4v_response(
                        convo_Q,
                        client = client,
                        deployment_name = deployment_name,
                        temperature = temperature
                    )

                    # Parse accuracy -- this line assumes it is integer, so will raise error if not
                    selected_category = int(raw_response_Q['choices'][0]['message']['content'])
                    print('Selected category:', selected_category)

                    # Check if accuracy is either 0 or 1 -- this line will raise error if not
                    assert selected_category in [0, 1]

                    # Determine accuracy
                    accuracy = 1 if selected_category == query_category else 0
                    print('Accuracy:', accuracy)

                    # Increase concept counter
                    unique_test_concepts[concept] += 1
                    # with open(f'results/{cotN}/concept_count.pkl', 'wb') as file:
                    #     pickle.dump(unique_test_concepts, file)  

                    # If code gets to this point, then while-loop condition can be set to false; no need to re-run API
                    break

                except openai.BadRequestError as bad_request_error_message:
                    print(bad_request_error_message)

                    selected_category = None
                    accuracy = None
                    raw_response_Q = bad_request_error_message
                    bad_request_error = True

                    break

                # KeyboardInterrupt does not trigger this except so you can kill python file when desired
                except Exception as e:
                    print(e)

                    # Exponential backoff with jitter; don't hammer API with calls
                    time.sleep(1 * 2 ** num_calls_to_API + random.uniform(0, 1))
                    print(f'Re-running API call. This is call {num_calls_to_API}.')
                    num_calls_to_API += 1

                    continue

            #########################################################################################################
            # Save information
            #########################################################################################################
            # How many API calls for that trial?
            print('Total number of API calls for this trial:', num_calls_to_API)
            
            elapsed_time = time.time() - start_time
            print('Elapsed time:', elapsed_time)

            # Add row for attempt information
            row = [
                cotN,
                test_condition,
                idx_test_item,

                concept,
                relation,
                object_,
                
                incontext_categories,
                num_positive,
                num_negative,
                query_category,
                convo_Q,

                selected_category,
                accuracy,
                raw_response_Q,
                bad_request_error,
                elapsed_time
            ]
            data.loc[len(data.index) + 1] = row

            # Re-save the full dataframe at each attempt
            save_path = f'results/{cotN}/{test_condition}_fewshot_results.csv'
            data.to_csv(save_path)


        #########################################################################################################
        # Next test item
        #########################################################################################################
        # Reset convo
        convo = [
            {"role": "system", "content": instruction_message}
        ]

        print()


