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


def update_convo(convo, incontext_image=None, incontext_message=None, idx_incontext=None, query_image=None, query_message=None, assistant_message=None):

    if incontext_image is not None:
        # Get category from filename
        incontext_image_category = get_category_from_filename(incontext_image)

        # Encode into data url
        incontext_image_data_url = local_image_to_data_url(incontext_image)

        # Update user message with correct image category
        idx_incontext_ordinal = append_ordinal_suffix(idx_incontext)
        formatted_incontext_message = incontext_message.format(category = incontext_image_category, idx_incontext_ordinal = idx_incontext_ordinal)

        dict_incontext_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": formatted_incontext_message},
                {"type": "image_url", "image_url": {"url": incontext_image_data_url, 'detail': 'high'}}
            ]
        }
        convo.append(dict_incontext_message)

    if query_image is not None:
        query_image_data_url = local_image_to_data_url(query_image)
        
        dict_query_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": query_message},
                {"type": "image_url", "image_url": {"url": query_image_data_url, 'detail': 'high'}}
            ]
        }
        convo.append(dict_query_message)

    if assistant_message is not None:
        dict_assistant_message = {
            "role": "assistant",
            "content": assistant_message
        }
        convo.append(dict_assistant_message)

    return convo

def append_ordinal_suffix(num):
    # Convert to string to easily access the last characters
    str_num = str(num)
    
    # Check for special cases: 11th, 12th, 13th
    if str_num.endswith("11") or str_num.endswith("12") or str_num.endswith("13"):
        return f"{num}th"
    
    # Determine the suffix based on the last digit
    last_digit = str_num[-1]
    
    if last_digit == '1':
        return f"{num}st"
    elif last_digit == '2':
        return f"{num}nd"
    elif last_digit == '3':
        return f"{num}rd"
    else:
        return f"{num}th"

def get_category_from_filename(filename):
    shortened_file_path = filename.split('/')[-1]
    category = int(shortened_file_path.split('_')[1])
    # exemplar = int(shortened_file_path.split('_')[-1][:-4])
    return category

def get_gpt4v_response(messages, client, deployment_name, temperature=0):
    '''Sends conversation history to API.
       The context window is set by providing the full conversation history as the 'messages' parameter. 
       GPT will not remember previous messages unless they are included in messages. Thus, we send whole conversation history as messages.
    '''

    # Send to API
    response = client.chat.completions.create(
        model = deployment_name,
        max_tokens = 2000,
        temperature = temperature,
        n = 1,
        # logprobs = True,  # not available for gpt-4-vision-preview
        messages = messages,
    )
    response_json = json.loads(response.json())
    return response_json

def local_image_to_data_url(image_path):
    ''' Function to encode a local image into data URL 
    '''
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def get_all_attempts(problem, total_incontexts, total_attempts):
    ''' Constructs dictionary of lists of file paths where the first num_incontext filepaths are in-context images
        and the last filepath (num_incontext + 1th) is the query image.

        A category has 1/2 chance being selected at every element, and a random file is selected from that category without replacement.
    '''

    # Filename construction
    base_directory = f"/svrt/results_problem_{problem}"
    filename_pattern = "{base_directory}/sample_{category}_{exemplar:04}.png"

    #
    dict_all_attempts = {num_incontext:
        {attempt: [] for attempt in range(1, total_attempts + 1)}
    for num_incontext in range(1, total_incontexts + 1)}

    #
    for num_incontext in dict_all_attempts.keys():

        # All possible exemplars, shuffled
        all_possible_exemplars = np.arange(1000)
        np.random.shuffle(all_possible_exemplars)

        # Iterate over each attempt number, and append to dictionary
        for attempt in range(total_attempts):

            # Get i to N+1 numbers in all_possible_exemplars, and turn into filenames
            list_exemplars = all_possible_exemplars[attempt*(num_incontext+1) : (attempt+1)*(num_incontext+1)].tolist()

            # Turn each into filename, randomly selecting either pos or neg category for each exemplar 
            list_filenames = [filename_pattern.format(
                base_directory=base_directory, 
                category=np.random.choice([0, 1]), 
                exemplar=exemplar
            ) for exemplar in list_exemplars]

            # Append
            dict_all_attempts[num_incontext][attempt + 1] = list_filenames
    
    return dict_all_attempts