from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import math
import pickle
from transformers import logging
import csv

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


def get_encodings(df, tokenizer, model):
    encodings = []
    for text in df:
        encoded_input = tokenizer(text,max_length=512, return_tensors='pt')

        output = model(**encoded_input)
        encodings.append(output.pooler_output.detach().numpy())
    # print(encodings)
    array = np.array(encodings)
    return array


def DataLoader(path_to_csv, batch_size):
    df = pd.read_csv(path_to_csv)
    df = df.iloc[:50000]
    df = df.drop(['PRODUCT_TYPE_ID'], axis=1)
    L = len(df.index)

    target_df = df.apply(lambda x: '%s %s %s' % (x['TITLE'], x['BULLET_POINTS'], x['DESCRIPTION']), axis=1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:

            limit = min(L, batch_end)

            X = get_encodings(target_df.iloc[batch_start:limit], tokenizer, model)

            product_ids = df['PRODUCT_ID'].iloc[batch_start:limit]
            yield (product_ids, X)

            batch_start += batch_size
            batch_end += batch_size



#load the model
Xgb_r = pickle.load(open("xgb_reg.pkl", "rb"))

test_datagen = DataLoader("test.csv",1)

#prediction
pred_list = []
id_list = []

for step in range(0,50000):
    ids, x = test_datagen.__next__()
    sx = np.squeeze(x, axis=1)

    y_pred = Xgb_r.predict(sx)
    pred_list.extend(y_pred)
    id_list.extend(ids)
    print(step)
    
# Combine PRODUCT_ID and prediction values into a Pandas DataFrame
result_df = pd.DataFrame({
    'PRODUCT_ID': id_list,
    'PRODUCT_LENGTH': pred_list
})

# Save the results to a CSV file
result_df.to_csv('predictions.csv', index=False)
