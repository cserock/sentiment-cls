import os, torch, sys, wandb
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, DebertaV2ForSequenceClassification

from packages.model.sentiment_classification.modules.metrics import compute_metrics
from packages.model.sentiment_classification.modules.trainer import CustomTrainer
from packages.model.sentiment_classification.modules.utils import load_yaml
from packages.model.sentiment_classification.modules.preprocess import preprocess_infer
from packages.model.sentiment_classification.modules.dataset import CustomDataset

# Root directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'inference_config.yaml')

config = load_yaml(config_path)

# Recorder directory
CHECKPOINT_DIR  = os.path.join(PROJECT_DIR, 'results', config.TEST.checkpoint_path)
OUTPUT_DIR = os.path.join(CHECKPOINT_DIR, 'test_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data directory
DATA_DIR = os.path.join(PROJECT_DIR, 'data', config.TEST.directory.dataset)

#DEVICE
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available() : device = torch.device('cuda')
elif torch.backends.mps.is_available() : device = torch.device('mps')
else : device=torch.device('cpu')
print(f'Using {device}')


def load_model(select_model, maxlen, loss, device):
    name = f'{select_model}_{maxlen}_{"".join(loss.split())}'
    output_dir = os.path.join(PROJECT_DIR, 'results', config.MODEL.results[name])
    os.makedirs(output_dir, exist_ok=True)
    pretrained_link = config.MODEL.pretrained_link[select_model]
    num_of_classes  = config.MODEL.num_of_classes
    checkpoint_path = output_dir

    print('=' * 50)
    print(f'Get Model & Tokenizer : {name}')
    print('=' * 50)
    test_args = TrainingArguments(output_dir=output_dir,
                                  dataloader_pin_memory=False,
                                  do_predict=True
                                  )
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_of_classes).to(device)

    trainer = CustomTrainer(model=model,
                            args=test_args,
                            compute_metrics=compute_metrics
                            )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_link)

    return trainer, tokenizer

def test():
    output_dir      = OUTPUT_DIR
    pretrained_link = config.MODEL.pretrained_link[config.MODEL.model_name]
    num_of_classes  = config.MODEL.num_of_classes
    max_seq_len     = config.MODEL.max_seq_len
    checkpoint_path = CHECKPOINT_DIR
    
    test_args = TrainingArguments(output_dir = output_dir,
                                      per_device_eval_batch_size = config.TEST.batch_size, 
                                      report_to = 'wandb',
                                      dataloader_pin_memory = False,
                                      do_eval = True,
                                      no_cuda=True
                                      )

    if config.MODEL.model_name == 'DeBERTa':
        model = DebertaV2ForSequenceClassification.from_pretrained(pretrained_link, num_labels=num_of_classes).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=num_of_classes).to(device)

    data = pd.read_csv(DATA_DIR + '_test.csv', index_col = 0)
    data = preprocess_infer(data)
    dataset = CustomDataset(data, pretrained_link, max_seq_len)

    wandb.login()
    trainer = CustomTrainer(model = model,
                            args = test_args,
                            compute_metrics = compute_metrics,
                            eval_dataset = dataset
                            )
    trainer.evaluate()

def classify(data, max_seq_len, trainer, tokenizer, device, topk = False, k = 3):
    print('=' * 50)
    print('Tokenizing...')
    print('=' * 50)

    items = list()

    if type(data) == str:
        item = {key: torch.tensor(val).to(device) for key, val in tokenizer(data,
                                                                            truncation = True,
                                                                            padding = 'max_length',
                                                                            max_length = max_seq_len).items()}
        items.append(item)

    elif type(data) == pd.DataFrame:
        for name in tqdm(data['sentence_r']):
            item = {key: torch.tensor(val).to(device) for key, val in tokenizer(name,
                                                                                truncation = True,
                                                                                padding = 'max_length',
                                                                                max_length = max_seq_len).items()}
            items.append(item)

    print('=' * 50)
    print('Predicting...')
    print('=' * 50)

    test_results = trainer.predict(items)
    top_val, top_ind = torch.topk(torch.nn.Softmax(dim = 1)(torch.FloatTensor(test_results.predictions)), k = k, dim = 1)
    top_val = top_val.detach().cpu().numpy()
    top_ind = top_ind.detach().cpu().numpy()
    max_val = top_val[:, 0]
    max_ind = top_ind[:, 0]
    if type(data) == str:
        if topk:
            return np.array(pd.DataFrame(top_ind).replace(config.LABELING.keys(), config.LABELING.values())), top_val
        else:
            return config.LABELING[max_ind[0]], max_val[0]

    elif type(data) == pd.DataFrame:
        data['template'] = max_ind
        data['template'] = data['template'].replace(config.LABELING.keys(), config.LABELING.values())
        return data, max_val
    
if __name__ == '__main__':
#     print('=' * 50)
#     print('Preprocessing...')
#     print('=' * 50)
# DataFrame
    # project_path = os.path.abspath(os.getcwd())
    # A = cls(preprocess_infer(pd.read_csv(f'{project_path}/model/data/cleaned_test.csv', index_col = 0)))
    # A.to_csv('inferenced.csv')
#     print(A)
#문장
    B = classify(preprocess_infer('텀블벅 프로젝트에 함께해 주셔서 고맙습니다.'))
    print(B)
    
# 테스트 사용
    # a = test()
    # print(a)
    pass