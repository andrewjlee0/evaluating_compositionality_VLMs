import base64
from mimetypes import guess_type
import numpy as np
import json
import base64
import httpx


def construct_convo(incontext_images, query_image, incontext_message, query_message):
    convo = []

    for idx_incontext, filename in enumerate(incontext_images):

        # Get category from filename
        incontext_image_category = get_category_from_filename(filename)

        idx_incontext_ordinal = append_ordinal_suffix(idx_incontext + 1)

        # Update user message with correct image category
        formatted_incontext_message = incontext_message.format(category = incontext_image_category, idx_incontext_ordinal = idx_incontext_ordinal)

        # Encode into data url
        with open(filename, "rb") as image_file:
            incontext_image_data = base64.b64encode(image_file.read()).decode("utf-8")

        dict_incontext_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": formatted_incontext_message},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": incontext_image_data
                }}
            ]
        }
        convo.append(dict_incontext_message)

    # Do similar steps above for query image
    with open(query_image, "rb") as image_file:
        query_image_data = base64.b64encode(image_file.read()).decode("utf-8")
    dict_query_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": query_message},
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": query_image_data
            }}
        ]
    }
    convo.append(dict_query_message)

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