### Model Directory Structure

python 3.11.11 설치
```bash
# python 3.11.11 설치
uv venv --python 3.11.11

# 활성화
source .venv/bin/activate

# 비활성화
deactivate

# 설치
uv pip install -r requirements.txt

```
<pre>
<code>
...
└── sentiment_classification
      ├── config
      │     ├── inference_config.yaml
      │     ├── preprocess_config.yaml
      │     ├── split_config.yaml
      │     └── train_config.yaml
      ├── data
      │     ├── cleaned_test.csv
      │     ├── cleaned_train.csv
      │     └── ko_sentence_template.csv
      ├── modules
      │     ├── dataset.py
      │     ├── losses.py
      │     ├── metrics.py
      │     ├── optimizer.py
      │     ├── preprocess.py
      │     ├── split.py
      │     ├── trainer.py
      │     └── utils.py
      ├── train.py
      ├── inference.py
      └── inference_streamlit.py
</code>
</pre>