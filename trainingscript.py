from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import xgboost as xg
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

def productLength(df):
    list_len = []
    for length in df:
        list_len.append(length)
    
    return np.array(list_len)


def DataLoader(path_to_csv,batch_size):
    df = pd.read_csv(path_to_csv)

    df = df.iloc[:1000] #running for the first 1000 points

    df = df.drop(['PRODUCT_ID','PRODUCT_TYPE_ID'],axis=1)
    L = len(df.index)

    target_df=df.apply(lambda x:'%s %s %s' % (x['TITLE'],x['BULLET_POINTS'],x['DESCRIPTION']),axis=1)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L :

            limit = min(L,batch_end)
            
            X = get_encodings(target_df.iloc[batch_start:limit])

            Y = productLength(df['PRODUCT_LENGTH'].iloc[batch_start:limit])

            yield(X,Y)

            batch_start += batch_size
            batch_end += batch_size


train_datagen = DataLoader("/content/drive/MyDrive/new_train.csv",3)

number_of_steps=1000//3

Xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)

for step in range(number_of_steps):
    x,y = train_datagen.__next__()
    sx = np.squeeze(x, axis=1)
    expanded_y = np.expand_dims(y, axis = 1)
    Xgb_r.fit(sx, expanded_y)


#Saving the model
file_name = "/content/drive/MyDrive/xgb_reg.pkl"

# save
pickle.dump(Xgb_r, open(file_name, "wb"))

#testing
actual_length = []
pred_length = []

for step in range(100):
    x,y = test_datagen.__next__()
    sx = np.squeeze(x, axis=1)
    length = y[0]

    actual_length.append(length)

    y_pred = xgb_r.predict(sx)

    pred_length.append(y_pred)

print(pred_length)

