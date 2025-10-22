import sys, os, torch
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
ROOT_DIR_MODEL = os.path.join(ROOT_DIR, 'model')
sys.path.append(ROOT_DIR)
sys.path.append(ROOT_DIR_MODEL)
import pandas as pd

from packages.model.sentiment_classification.classify import load_model, classify
from packages.model.sentiment_classification.modules.preprocess import preprocess_infer

if torch.cuda.is_available() : device = torch.device('cuda')
elif torch.backends.mps.is_available() : device = torch.device('mps')
else : device=torch.device('cpu')
print(f'Using {device}')

MODEL, TOKENIZER = load_model('KoBERT', 128, 'Cross Entropy', 'cpu')
MODEL_DICT = {'KoBERT_128_CrossEntropy': MODEL}
TOKEN_DICT = {'KoBERT_128_CrossEntropy': TOKENIZER}
MAX_SEQ_LEN = 128

K = 3

class ClassifySentimentService:
    def __init__(self):
        super().__init__()

    def classify(self, sentence):
        print('=' * 50)
        print(f'max_seq_len : {MAX_SEQ_LEN}')
        print('=' * 50)

        cate, prob = classify(preprocess_infer(sentence), MAX_SEQ_LEN, MODEL, TOKENIZER, device, True, K)
        result = pd.DataFrame({'template': cate.flatten(), 'prob': prob.flatten()})
        result_json = result.to_json(orient='records')
        return result_json

    def classify_csv(self, df_json):        # max_seq_len = int(model_name.split('_')[1])
        df = pd.read_json(df_json)
        df.reset_index(drop=True, inplace=True)
        inferdf, _ = classify(preprocess_infer(df),MAX_SEQ_LEN, MODEL, TOKENIZER, device, True, K)
        inferdf_json = inferdf.to_json(orient='records')
        return inferdf_json
