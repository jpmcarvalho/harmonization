import os
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Harmonization of data")
parser.add_argument('--similarity_threshold', type=float, default=0.7, help='Similarity threshold (default: 0.7)')
parser.add_argument('--file_name', type=str, default='Ireland', help='Name of the file to be harmonized')
args = parser.parse_args()

# model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "sentence-transformers/bert-base-nli-mean-tokens"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze()  # Take the [CLS] embedding token


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_closest_word(word, my_dict):
    # Word embedding
    embedding_word = get_embedding(word).numpy()

    # closest_distance = float('inf')
    higher_sim = -np.inf
    closest_word = None

    for item in my_dict:

        embedding_item = get_embedding(item).numpy()
        # distance = np.linalg.norm(embedding_word - embedding_item)
        sim = cos_sim(embedding_word, embedding_item)
        if higher_sim < abs(sim):
            higher_sim = sim
            closest_word = item

    if higher_sim < 0.7:
        closest_word = 'nan'

    return closest_word


# read the data
# name_file = 'Slovenia.csv'
# name_file = 'Ireland.csv'
name_file = f'{args.file_name}.csv'
data = pd.read_csv(f'data/raw_data/{name_file}')

table_red_cap = pd.read_csv('data/DEPASSFamily_DataDictionary_2024-10-09.csv')
id_code = table_red_cap['Variable / Field Name'].tolist()

if 'family' in data.columns[0].lower():
    # add new column at the beginning equals to the first column
    data.insert(0, 'participant_id', data.iloc[:, 0])

# remove last two columns
if (len(data.columns) - len(id_code)) != 0:
    data = data.iloc[:, : -(len(data.columns) - len(id_code))]

# rename the columns
data.columns = id_code

# delete the rows that contains all nan values
data = data.dropna(how='all')

# convert all the columns to string values
data = data.astype(str)

# replace 'removed' value to 'nan'
data = data.map(lambda x: 'nan' if 'remove' in x.lower() else x)

# dict to field type
field_type = table_red_cap[['Variable / Field Name', 'Field Type']]
field_type = field_type.set_index('Variable / Field Name').T.to_dict('records')[0]

# filtrer the table_red_cap with Field Type = text
# table_red_cap_radio = table_red_cap[table_red_cap['Field Type'] == 'radio']

# iterate over the columns
pd.set_option('future.no_silent_downcasting', True)
res_pd = pd.DataFrame()
for col in data.columns:

    if '_dob' in col:
        tmp = data[col].str.strip()
        for row in range(len(data[col])):
            if not pd.isnull(data[col][row]) and data[col][row] != 'nan':
                # check if value can be a date with format yyyy-mm-dd
                try:
                    tmp[row] = pd.to_datetime(tmp[row], format='%d/%m/%Y').strftime('%Y-%m-%d')
                except:
                    tmp[row] = 'nan'
                pass
        res_pd[col] = tmp
        pass

    elif field_type[col] == 'yesno':
        tmp = data[col].str.strip()
        dict_values = {'yes': 1, 'no': 0}

        for row in range(len(data[col])):
            if not pd.isnull(data[col][row]) and data[col][row] != 'nan':
                # if get error try to find the closed string dict key
                if tmp[row].lower() not in dict_values.keys():

                    value = tmp[row].lower()
                    # get the closed string
                    closed_string = get_closest_word(value, dict_values)
                    # get the closest string value using large language model

                    # replace the value
                    if closed_string == 'nan':
                        tmp[row] = 'nan'
                    else:
                        tmp[row] = dict_values[closed_string]
                else:
                    tmp[row] = dict_values[tmp[row].lower()]
        res_pd[col] = tmp

    elif field_type[col] == 'radio':
        # get the values of the column
        values = \
            table_red_cap[table_red_cap['Variable / Field Name'] == col][
                'Choices, Calculations, OR Slider Labels'].values[
                0]
        values = values.split('|')

        dict_values = {}
        for x in values:
            dict_values[x.split(', ')[1].strip().lower()] = int(x.split(', ')[0])

        # replace the values but strip them first
        tmp = data[col].str.strip()
        for row in range(len(data[col])):
            if not pd.isnull(data[col][row]) and data[col][row] != 'nan':

                # if get error try to find the closed string dict key
                if tmp[row].lower() not in dict_values.keys():

                    value = tmp[row].lower()
                    # get the closed string
                    closed_string = get_closest_word(value, dict_values)
                    # get the closest string value using large language model

                    # replace the value
                    if closed_string == 'nan':
                        tmp[row] = 'nan'
                    else:
                        tmp[row] = dict_values[closed_string]
                else:
                    tmp[row] = dict_values[tmp[row].lower()]

        res_pd[col] = tmp

    else:
        res_pd[col] = data[col]

# replace 'nan' to np.nan
res_pd = res_pd.replace('nan', np.nan)

# save the data
os.makedirs('data/res/', 0o777, True)
res_pd.to_csv(f'data/res/{name_file}', index=False)
