from tqdm import tqdm
import pandas as pd
import re

'''
INPUT
-Train : df 형식은 pd.DataFrame / column은 sentence, template
-Inference : df 형식은 pd.DataFrame / column은 sentence이 있어야 함
'''

def duplicate_count(df):
    check = {}
    df_drop = df.drop_duplicates(keep = 'first')
    df_count = df_drop['sentence'].value_counts()
    for name in tqdm(df_count[df_count >= 2].index):
        cate = df[df['sentence'] == name]['template'].value_counts()
        check[name] = cate.index[0]
    return check

def drop_duplicate_by_count(df, check):
    df_drop = df.copy()
    for key, val in tqdm(check.items()):
        df_drop.loc[df_drop['sentence'] == key, 'template'] = val
    df_drop = df_drop.drop_duplicates(keep = 'first').reset_index(drop = True)
    return df_drop

def regularize(df, cased = False):  ### cased = True -> 대소문자 구별 / cased = False -> 영어는 모두 소문자
    if type(df) == pd.DataFrame:
        regularized = []
        for word in tqdm(df['sentence']):
            regularized.append(re.sub("[^가-힣a-zA-Z0-9]", " ", word))

        df['sentence_r'] = regularized
        df['sentence_r'] = df['sentence_r'].str.strip()

        for i in range(10):
            df['sentence_r'] = df['sentence_r'].str.replace('  ',' ')
            df['sentence_r'] = df['sentence_r'].str.replace('   ',' ')
        
        idx = []
        for c in range(50):
            idx.append(list(df[df['sentence_r'] == ' '* c].index))
        idx = sum(idx, [])
        df_ = df.drop(idx).reset_index(drop = True)

        if not cased:
            df_['sentence_r'] = df_['sentence_r'].str.lower()

        return df_
    
    elif type(df) == str:
        word = re.sub("[^가-힣a-zA-Z0-9]", " ", df)
        word = word.strip()
        if word == '':
            raise Exception('Cannot Preprocessing!!!')
        for i in range(10):
            word = word.replace('  ',' ')
            word = word.replace('   ',' ')
        return word
        

def preprocess_train(df):
    df.dropna(how = 'any', inplace = True)
    df.reset_index(drop = True, inplace = True)
    check = duplicate_count(df)
    df_ = df.drop_duplicates(keep = 'fisrt').reset_index(drop = True)
    df_drop = drop_duplicate_by_count(df_, check)
    df_regularized = regularize(df_drop)
    df_regularized = df_regularized[['sentence','sentence_r','template']]
    df_regularized = df_regularized.drop_duplicates(keep = 'first')
    df_regularized_ = df_regularized.drop_duplicates(subset = ['sentence_r'], keep = False).reset_index(drop = True)
    return df_regularized_

def preprocess_infer(df):
    if type(df) == pd.DataFrame:
        df.dropna(how = 'any', inplace = True)
        return regularize(df)
    elif type(df) == str:
        return regularize(df)

'''
OUTPUT
-Train : 형식은 pd.DataFrame / column은 sentence, sentence_r, template
-Inference : 형식은 pd.DataFrame / column은 기존 column + sentence_r
'''

if __name__ == '__main__':
# preprocess for training
    # df_ = preprocess_train(df)
    # df_.to_csv(path)
# preprocess for inference
    # import pandas as pd
    # a = preprocess_infer(pd.read_csv('/VOLUME/py_model/data/cleaned_test.csv', index_col = 0))
    pass
