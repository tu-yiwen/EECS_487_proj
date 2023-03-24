import pandas as pd

def str2label(df, split, name):
    #print('Converting string to label...')
    # store the labels into a new dataframe
    df_label = df
    for n, row in df.iterrows():
        if row["overall_sentiment"] == "neutral":
            df_label["overall_sentiment"][n] = 1
        elif row["overall_sentiment"] == "negative" or row["overall_sentiment"] == "very_negative":
            df_label["overall_sentiment"][n] = 0
        else:
            df_label["overall_sentiment"][n] = 2
        
        if split == "binary":
            if row["humour"] == "not_funny":
                df_label["humour"][n] = 0
            else:
                df_label["humour"][n] = 1
            if row["sarcastic"] == "not_sarcastic":
                df_label["sarcastic"][n] = 0
            else:
                df_label["sarcastic"][n] = 1
            if row["offensive"] == "not_offensive":
                df_label["offensive"][n] = 0
            else:
                df_label["offensive"][n] = 1
            if row["motivational"] == "not_motivational":
                df_label["motivational"][n] = 0
            else:
                df_label["motivational"][n] = 1
        elif split == "multi":
            if row["humour"] == "not_funny":
                df_label["humour"][n] = 0
            elif row["humour"] == "funny":
                df_label["humour"][n] = 1
            elif row["humour"] == "very_funny":
                df_label["humour"][n] = 2
            else:
                df_label["humour"][n] = 3
            if row["sarcastic"] == "not_sarcastic":
                df_label["sarcastic"][n] = 0
            elif row["sarcastic"] == "little_sarcastic":
                df_label["sarcastic"][n] = 1
            elif row["sarcastic"] == "very_sarcastic":
                df_label["sarcastic"][n] = 2
            else:
                df_label["sarcastic"][n] = 3
            if row["offensive"] == "not_offensive":
                df_label["offensive"][n] = 0
            elif row["offensive"] == "slight":
                df_label["offensive"][n] = 1
            elif row["offensive"] == "very_offensive":
                df_label["offensive"][n] = 2
            else:
                df_label["offensive"][n] = 3
            if row["motivational"] == "not_motivational":
                df_label["motivational"][n] = 0
            else:
                df_label["motivational"][n] = 1
                
    # store the new dataframe into a pickle file
    df_label.to_pickle(name+"_label")