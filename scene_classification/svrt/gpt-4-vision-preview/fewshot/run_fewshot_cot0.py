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

from utils_fewshot_cot0 import *


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
    base_url = f'{api_base}/openai/deployments/{deployment_name}'  # for some reason, changing this to base_url fixes a problem where GPT does not want to describe images
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
'''

incontext_message = '''Here is the {idx_incontext_ordinal} image. This image belongs to category {category}.'''

query_message = '''Here is the target image. Does this image belong to category 0 or 1? Please respond with either "0" or "1" and only one of these two numbers. You must provide a final answer, even if you are uncertain.'''


#########################################################################################################
# Setup
#########################################################################################################
np.random.seed(3)

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

# conditions = ['chain_of_thought0', 'chain_of_thought1', 'chain_of_thought2', 'chain_of_thought3']
cotN = 'chain_of_thought0'
range_problems = range(1, total_problems + 1)
range_incontexts = range(1, total_incontexts + 1)
range_attempts = range(1, total_attempts + 1)


#########################################################################################################
# Run
#########################################################################################################
for problem in range_problems:

    # Generate images for all attempts
    dict_all_attempts = get_all_attempts(
        problem, 
        total_incontexts, 
        total_attempts
    )
    # keys = attempt idx, values = list of length total_attempts + 1

    # Save structure of all attempts
    with open(f'results/{cotN}/{problem}_dict_all_attempts.pkl', 'wb') as file:
        pickle.dump(dict_all_attempts, file)

    if problem == 1:
        data = pd.read_csv(f'results/{cotN}/23_fewshot_results.csv')
        data = data.drop(columns = ['Unnamed: 0'])
        range_incontexts = range(4, total_incontexts + 1)
        range_attempts = range(1, 5)
    elif problem > 1:
        continue
        # range_incontexts = range(1, total_incontexts + 1)
        # range_attempts = range(1, total_attempts + 1)


    #########################################################################################################
    # Iterate over all 10 number of in-contexts and attempts
    #########################################################################################################
    for num_incontext in range_incontexts:
        for attempt in range_attempts:  
            print('Condition:', cotN)
            print('Problem:', problem)
            print('Number of in-context examples:', num_incontext)
            print('Attempt:', attempt)

            # Equivalent to MATLAB's tic
            start_time = time.time()

            # Get 
            incontext_and_query_images = dict_all_attempts[num_incontext][attempt]

            # In-context image filenames are the first num_incontext, the query image filename is num_incontext+1 (last)
            incontext_images = incontext_and_query_images[:num_incontext]
            query_image = incontext_and_query_images[-1]

            # Query category ground-truth
            incontext_categories = [get_category_from_filename(incontext_image) for incontext_image in incontext_images]
            query_category = get_category_from_filename(query_image)

            # How many pos and neg in incontext images
            num_positive = np.sum(incontext_categories)
            num_negative = len(incontext_images) - np.sum(incontext_categories)

            # Get convo
            convo = construct_convo(
                incontext_images, 
                query_image,
                instruction_message, 
                incontext_message, 
                query_message
            )
            
            # Print everything
            print('In-context and query images:', incontext_and_query_images)
            print('In-context images:', incontext_images)
            print('Query image:', query_image)
            print('In-context categories:', incontext_categories)
            print('Query category:', query_category)
            print('Number of positive:', num_positive)
            print('Number of negative:', num_negative)
            # print('Convo:', convo)


            #########################################################################################################
            # Interact with GPT-4 
            #########################################################################################################
            # Ensure that trial response is a single number, 0 or 1
            response_inappropriate = True
            num_calls_to_API = 1
            while response_inappropriate:

                # Keep calling API until it works or GPT's response format is appropriate
                try:

                    # Get GPT-4V's response
                    temperature = 0
                    raw_response = get_gpt4v_response(
                        convo,
                        client = client,
                        deployment_name = deployment_name,
                        temperature = temperature
                    )
                    # print("GPT-4V's Raw Response to Trial:", raw_response_to_trial)

                    # Parse accuracy -- this line assumes it is integer, so will raise error if not
                    selected_category = int(raw_response['choices'][0]['message']['content'])
                    print('Selected category:', selected_category)

                    # Check if accuracy is either 0 or 1 -- this line will raise error if not
                    assert selected_category in [0, 1]

                    # Determine accuracy
                    accuracy = 1 if selected_category == query_category else 0
                    print('Accuracy:', accuracy)

                    # If code gets to this point, then while-loop condition can be set to false; no need to re-run API
                    response_inappropriate = False

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
                problem,
                num_incontext,
                attempt,
                
                incontext_images,
                query_image,
                query_category,
                num_positive,
                num_negative,
                convo,

                selected_category,
                accuracy,
                raw_response,
                elapsed_time
            ]
            data.loc[len(data.index) + 1] = row

            # Re-save the full dataframe at each attempt
            save_path = f'results/{cotN}/{problem}_fewshot_results.csv'
            data.to_csv(save_path)

            print()


