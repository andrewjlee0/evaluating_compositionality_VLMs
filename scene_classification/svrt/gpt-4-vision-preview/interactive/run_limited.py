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

from utils_limited import *


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
system_message = '''
Today, you'll be doing a visual reasoning task. 
In this task, there are 23 problems. 
Each problem consists of a series of images. 
Each image belongs to one of two categories (0 or 1). 
Your job is to learn to correctly classify the images of each problem.
On a trial, you will be given an image and asked to categorize it.
You will respond with either "0" or "1" and only one of these two numbers.
Then, you will receive feedback for your response ("Correct!" or "Incorrect!").
Please reply to this feedback with "Feedback received." and only this reply.
Don't worry if you're wrong the first few times. That's normal.
'''

convo_history = [
    {"role": "system", "content": system_message}
]
trial_prompt = '''Does this image belong to category 1 or 0? Please respond with either "0" or "1" and only one of these two numbers. You must provide a final answer, even if you are uncertain.'''

# All trial data
columns = [
    'trial',
    'file_path',
    'shortened_file_path',
    'problem',
    'category',
    'exemplar',
    'num_calls_to_API',
    'raw_response_to_trial',
    # 'logprobs_raw_response_to_trial',
    'selected_category',
    'accuracy',
    'correct_in_row',
    'feedback',
    'raw_response_to_feedback',
    # 'logprobs_raw_response_to_feedback',
    'feedback_response',
    'raw_response_to_next_problem',
    'next_problem_response',
    'temperature',
]
data = pd.DataFrame(columns=columns)


#########################################################################################################
# Randomized variables
#########################################################################################################
np.random.seed(3)

# Straightforward problem sequence
problem_sequence = list(range(1,24))

# Dictionary of trial sequences by problem
randomized_trial_sequence_dict = {
    problem:None for problem in problem_sequence
}

for problem in problem_sequence:
    # Construct list of file paths that map onto a randomized trial sequence
    # Append to dictionary
    randomized_trial_sequence = get_randomized_trial_sequence(problem)
    randomized_trial_sequence_dict[problem] = randomized_trial_sequence

# Save randomized trial sequence
with open('results/limited/randomized_trial_sequence_dict.pkl', 'wb') as file:
    pickle.dump(randomized_trial_sequence_dict, file)


#########################################################################################################
# Run
#########################################################################################################
# Iterate over problems
for problem in problem_sequence:

    # Tracker for consecutive correct answers
    correct_in_row = 0   # switch problems if =7


    # Iterate over each file path (iterate over the trial sequence)
    randomized_trial_sequence = randomized_trial_sequence_dict[problem]
    for trial, file_path in enumerate(randomized_trial_sequence, 1):

        #########################################################################################################
        # Start trial
        #########################################################################################################
        # Equivalent to MATLAB's tic
        start_time = time.time()

        # Obtain and print information about this trial
        shortened_file_path = file_path.split('/')[-1]
        category = int(shortened_file_path.split('_')[1])
        exemplar = int(shortened_file_path.split('_')[-1][:-4])

        print('Problem:', problem)
        print('Trial:', trial)
        print('File path:', file_path)
        print('Shortened file path:', shortened_file_path)
        print('Category:', category)
        print('Exemplar:', exemplar)

        # Load image
        image_data_url = local_image_to_data_url(file_path)

        # Update convo history with trial prompt
        convo_history_TP = update_convo_history(convo_history, user_message=trial_prompt, image_data_url=image_data_url)
        # print('Conversation history + trial prompt:')
        # for message in convo_history_TP:
        #     print(message)
        

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
                raw_response_to_trial = get_gpt4v_response(
                    convo_history_TP, 
                    temperature=temperature,
                    client = client,
                    deployment_name = deployment_name
                )
                # print("GPT-4V's Raw Response to Trial:", raw_response_to_trial)

                # Parse accuracy -- this line assumes it is integer, so will raise error if not
                selected_category = int(raw_response_to_trial['choices'][0]['message']['content'])
                print('Selected category:', selected_category)

                # Check if accuracy is either 0 or 1 -- this line will raise error if not
                assert selected_category in [0, 1]

                # Determine accuracy
                accuracy = 1 if selected_category == category else 0
                print('Accuracy:', accuracy)

                # Determine logprobs of response to trial prompt (selected_category)
                # logprobs_raw_response_to_trial = int(raw_response_to_trial['choices'][0]['logprobs'])

                # Determine feedback
                feedback = f'''{'Correct!' if accuracy == 1 else 'Incorrect!'} This image belongs to category {category}. Please reply to this message with "Feedback received." and only this reply.'''
                print('Feedback:', feedback)

                # Construct messages including feedback
                convo_history_TP_TR_FB = update_convo_history(
                    convo_history_TP,
                    user_message = feedback,
                    assistant_message = str(selected_category)
                )
                # print('Conversation history + trial prompt + trial response + feedback:')
                # for message in convo_history_TP_TR_FB:
                #     print(message)

                # Give feedback
                raw_response_to_feedback = get_gpt4v_response(
                    convo_history_TP_TR_FB, 
                    temperature=temperature,
                    client = client,
                    deployment_name = deployment_name
                )
                # print("GPT-4V's Raw Response to Feedback:", raw_response_to_feedback)

                # Determine response to feedback
                feedback_response = raw_response_to_feedback['choices'][0]['message']['content']
                print("GPT-4V's Response to Feedback:", feedback_response)

                # Check if response to feedback is as instructed -- this line will raise error if not
                assert feedback_response == 'Feedback received.'

                # Determine logprobs of response to feedback ('Feedback received.')
                # logprobs_raw_response_to_feedback = int(raw_response_to_feedback['choices'][0]['logprobs'])

                # Update convo history with GPT's response to feedback
                convo_history_TP_TR_FB_FBR = update_convo_history(
                    convo_history_TP_TR_FB,
                    assistant_message = feedback_response
                )
                print('Convo history + trial prompt + trial response + feedback + feedback response:')
                for message in convo_history_TP_TR_FB_FBR:
                    print(message)

                # Conversation history can be set to current list of messages
                convo_history = convo_history_TP_TR_FB_FBR

                # Save convo history
                with open(f'results/limited/{problem}_convo_history.pkl', 'wb') as file:
                    pickle.dump(convo_history, file)

                # If code gets to this point, then while-loop condition can be set to false; no need to re-run API
                response_inappropriate = False

            # KeyboardInterrupt does not trigger this except so you can kill python file when desired
            except Exception as e:
                print(e)

                # Exponential backoff with jitter; don't hammer API with calls
                time.sleep(1 * 2 ** num_calls_to_API + random.uniform(0, 1))
                print('Re-running API call.')
                num_calls_to_API += 1

                continue


        #########################################################################################################
        # Save information
        #########################################################################################################
        # How many API calls for that trial?
        print('Total number of API calls for this trial:', num_calls_to_API)

        elapsed_time = time.time() - start_time
        print('Elapsed time:', elapsed_time)

        # Determine how many correct in row so far
        if accuracy == 1:
            correct_in_row += 1 
        else:
            correct_in_row = 0
        print('Correct in a row:', correct_in_row)

        # Add row for trial information
        row = [
            trial, 
            file_path, 
            shortened_file_path, 
            problem, 
            category, 
            exemplar, 
            num_calls_to_API,
            raw_response_to_trial, 
            # logprobs_raw_response_to_trial,
            selected_category, 
            accuracy, 
            correct_in_row, 
            feedback,
            raw_response_to_feedback,
            # logprobs_raw_response_to_feedback,
            feedback_response,
            np.nan,
            np.nan,
            temperature
            ]
        data.loc[len(data.index) + 1] = row

        # Save the full dataframe at each trial, rather than at each problem or until very end
        save_path = f'results/limited/{problem}_limited_results.csv'
        data.to_csv(save_path)

        print()


        #########################################################################################################
        # Next problem
        #########################################################################################################
        if correct_in_row == 7 or trial == 34:

            # Reset correct in row counter
            correct_in_row = 0

            # VERY IMPORTANT: reset convo history, treating each problem as separate context windows
            convo_history = [
                {"role": "system", "content": system_message}
            ]

            # # Also reset the results
            # data = pd.DataFrame(columns=columns)

            break  # go to next problem, skip rest of trials
    
