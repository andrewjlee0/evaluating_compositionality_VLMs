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


def update_convo_history(convo_history, user_message=None, assistant_message=None, image_data_url=None):

    # Append assistant's message if available (GPT's response)
    if assistant_message is not None:
        convo_history.append(
            {"role": "assistant", "content": assistant_message}
        )
    
    # Construct user message if available
    if user_message is not None:
        new_user_message = {
            "role": "user", 
            "content": [
                {"type": "text", "text": user_message}
            ]
        }
        # Append image to user message if available
        if image_data_url is not None:
            new_user_message['content'].append(
                {"type": "image_url", "image_url": {"url": image_data_url, 'detail': 'high'}}
            )
        # Append user message to conversation history
        convo_history.append(new_user_message)

    # Remove all messages before the last ten, except system message
    shaved_convo_history = shave_convo_history(convo_history)
    return shaved_convo_history

def shave_convo_history(convo_history):
    ''' Makes sure that conversation history contains ONLY the most recent 10 trials / images, plus system message
        All previous messages are gone, except the system message.
    '''

    shaved_convo_history = []

    # Ignore the first message (system message) and reverse the list
    reversed_convo_history = reversed(convo_history[1:])

    # Iterate *backwards* over each message in convo history
    counter = 0
    for message in reversed_convo_history:

        message_copy = message.copy()
        role = message_copy['role']
        contents = message_copy['content']

        # Only check message if it is user message, which can contain images
        if role == 'user':

            # Look for 'image_url' in message contents
            for content in contents:

                if content['type'] == 'image_url':
                    counter += 1

        # Append each message
        shaved_convo_history.append(message)

        # Stop appending if counter greater than or equal to 10
        # This assumes that each trial starts as the image message, which is true
        if counter >= 10:
            break
    
    # Append system message at last
    system_message = convo_history[0]
    shaved_convo_history.append(system_message)

    return list(reversed(shaved_convo_history))

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

def get_randomized_trial_sequence(problem):
    ''' Constructs list of file paths that map onto a randomized trial sequence, such that
        a category has 1/2 chance being selected at each trial, and a random file is selected from that category without replacement.
    '''

    # Base directory and filename pattern
    base_directory = f"/svrt/results_problem_{problem}"
    filename_pattern = "sample_{category}_00{index:02}.png"

    # Number of files and categories
    num_files = 17  # From 0000 to 0016
    categories = [0, 1]

    # Generate all filenames for each category and shuffle them
    files_in_category = {
        category: [filename_pattern.format(category=category, index=i) for i in range(num_files)]
        for category in categories
    }

    # Shuffle each category's filenames
    np.random.shuffle(files_in_category[0])
    np.random.shuffle(files_in_category[1])

    # Prepare to select all files using a balanced approach
    randomized_trial_sequence = []
    remaining_files = {0: num_files, 1: num_files}  # Tracks remaining files in each category
    total_files = num_files * len(categories)  # Total files to select

    while total_files > 0:
        # Choose a category that still has files left
        available_categories = [cat for cat, count in remaining_files.items() if count > 0]
        chosen_category = np.random.choice(available_categories)
        
        # Select the next file from the chosen category
        chosen_file = files_in_category[chosen_category].pop()
        full_path = f"{base_directory}/{chosen_file}"
        randomized_trial_sequence.append(full_path)
        
        # Update remaining count and total
        remaining_files[chosen_category] -= 1
        total_files -= 1
    
    return randomized_trial_sequence