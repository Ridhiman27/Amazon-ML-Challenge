from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import math
import pickle

def get_encodings(df):
    encodings = []
    for text in df:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained("bert-base-uncased")

        encoded_input = tokenizer(text,max_length=512, return_tensors='pt')

        output = model(**encoded_input)
        encodings.append(output.pooler_output.detach().numpy())
    # print(encodings)
    array = np.array(encodings)
    return array


def DataLoader(path_to_csv,batch_size):
    df = pd.read_csv(path_to_csv)

    df = df.iloc[:1200] #TODO:change it according to the range you are running for

    df = df.drop(['PRODUCT_ID','PRODUCT_TYPE_ID'],axis=1)
    L = len(df.index)

    target_df=df.apply(lambda x:'%s %s %s' % (x['TITLE'],x['BULLET_POINTS'],x['DESCRIPTION']),axis=1)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L :

            limit = min(L,batch_end)
            
            X = get_encodings(target_df.iloc[batch_start:limit])


            yield(X)

            batch_start += batch_size
            batch_end += batch_size

#load the model
Xgb_r = pickle.load(open("/content/drive/MyDrive/xgb_reg.pkl", "rb"))

test_datagen = DataLoader("/content/drive/MyDrive/test.csv",1)

#prediction
pred_list = []
for step in range(2544,10000,32):
    x = test_datagen.__next__()
    sx = np.squeeze(x, axis=1)

    # actual_length.append(length)

    y_pred = Xgb_r.predict(sx)
    print(y_pred)
    print(y_pred[0])

    pred_list.extend(y_pred)
    print(step)
    




df = pd.DataFrame(pred_list, columns=['prediction'])

#check this code once not sure
csv = df.reset_index().rename(columns={'index': 'id', 'prediction': 'prediction'})

csv.to_csv("/content/drive/MyDrive/1008-2544.csv", index=False)

