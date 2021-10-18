# -*- coding: utf-8 -*-
"""04-gpt2-sentiment-ppo-training.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Hg6lwG32aBluimPUNZeElfwpjRTjYzaL

# Tune GPT2 to generate positive reviews
> Optimise GPT2 to produce positive IMDB movie reviews using a BERT sentiment classifier for rewards.

<div style="text-align: center">
<img src='images/gpt2_bert_training.png' width='600'>
<p style="text-align: center;"> <b>Figure:</b> Experiment setup to tune GPT2. The yellow arrows are outside the scope of this notebook, but the trained models are available through Hugging Face. </p>
</div>


In this notebook we fine-tune GPT2 (small) to generate positive movie reviews based on the IMDB dataset. The model gets 5 tokens from a real review and is tasked to produce positive continuations. To reward positive continuations we use a BERT classifier to analyse the sentiment of the produced sentences and use the classifier's outputs as rewards signals for PPO training.

## Setup experiment

### Import dependencies
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext autoreload
# %autoreload 2

import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
tqdm.pandas()

from transformers import GPT2Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from trl.gpt2 import GPT2HeadWithValueModel, respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt

"""### Configuration"""

config = {
    "lm_name": "lvwerra/gpt2-imdb",
    "ref_lm_name": "lvwerra/gpt2-imdb",
    "cls_model_name": "lvwerra/bert-imdb",
    "tk_name": "gpt2",
    "steps": 25600,
    "batch_size": 256,
    "forward_batch_size": 16,
    "ppo_epochs": 4,   
    "txt_in_len": 5,
    "txt_out_len": 15,
    "lr": 1.41e-5,
    "init_kl_coef":0.2,
    "target": 6,
    "horizon":10000,
    "gamma":1,
    "lam":0.95,
    "cliprange": .2,
    "cliprange_value":.2,
    "vf_coef":.1, 
}

"""You can see that we load a GPT2 model called `gpt2_imdb`. This model was additionally fine-tuned on the IMDB dataset for 1 epoch with the huggingface [script](https://github.com/huggingface/transformers/blob/master/examples/run_language_modeling.py) (no special settings). The other parameters are mostly taken from the original paper ["Fine-Tuning Language Models from Human Preferences"](
https://arxiv.org/pdf/1909.08593.pdf). This model as well as the BERT model is available in the Huggingface model zoo [here](https://huggingface.co/models). The following code should automatically download the models.

### Initialize W&B logger
We use `wandb`to log all the metrics during training.
"""

wandb.init(name='run-42', project='gpt2-sentiment', config=config)

"""## Load data and models

### Load IMDB dataset
The IMDB dataset contains 50k movie review annotated with "positive"/"negative" feedback indicating the sentiment. It can be downloaded from Kaggle ([link](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)).  We load the IMDB dataset into a DataFrame and filter for comments that are at least 500 characters long and take the first 1000 characters of each comment. The first filter we apply to avoid comments that are less than `txt_in_len` token long and the second to avoid tokenizing way more text than we actually need.
"""

# makes sure you download the imdb-dataset in the data folder
df = pd.read_csv('../data/imdb-dataset.csv')

# make sure the comments are long enough
df = df.loc[df['review'].str.len() > 500]

# make sure comments are not too long
df['review'] = df['review'].apply(lambda x: x[:1000])

df.head()

"""### Load BERT classifier
We load a BERT classifier fine-tuned on the IMDB dataset.
"""

sentiment_model = AutoModelForSequenceClassification.from_pretrained(config["cls_model_name"])
sentiment_tokenizer = AutoTokenizer.from_pretrained(config["cls_model_name"])

"""The model outputs are the logits for the negative and positive class. We will use the logits for positive class as a reward signal for the language model."""

text = 'this movie was really bad!!'
output = sentiment_model.forward(sentiment_tokenizer.encode(text, return_tensors="pt"))
output

text = 'this movie was really good!!'
output = sentiment_model.forward(sentiment_tokenizer.encode(text, return_tensors="pt"))
output

"""The resulting reward signal:"""

output[0][0, 1]

"""### Load pre-trained GPT2 language models

We load the GPT2 model with a value head and the tokenizer. We load the model twice; the first model is optimized while the second model serves as a reference to calculate the KL-divergence from the starting point. This serves as an additional reward signal in the PPO training to make sure the optimized model does not deviate too much from the original language model.
"""

gpt2_model = GPT2HeadWithValueModel.from_pretrained(config['lm_name'])
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(config['ref_lm_name'])
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(config['tk_name'])

"""### Watch model with wandb
This wandb magic logs the gradients and weights of the model during training.
"""

wandb.watch(gpt2_model, log='all')

"""### Move models to GPU

If `cuda` is available move the computations to the GPU.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ = gpt2_model.to(device)
_ = sentiment_model.to(device)
_ = gpt2_model_ref.to(device)

"""### Tokenize IMDB reviews

We tokenize all IMDB in advance to avoid tokenizing twice. In the first step we encode the queries and slice the first `txt_in_len` tokens. In a second step we decode these tokens back to text for later display.
"""

df['tokens'] = df['review'].progress_apply(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :config['txt_in_len']])

df['query'] = df['tokens'].progress_apply(lambda x: gpt2_tokenizer.decode(x))

"""## Optimize model

**Steps**

The training loop consists of the following steps:
1. Get a batch of queries
2. Get the query responses from the policy
3. Join query and responses and tokenize for BERT analysis
4. Get sentiments for query/responses from BERT
5. Optimize policy with PPO using the (query, response, reward) triplet
6. Log all the training statistics

**Forward batching**

Since the models can be fairly big and we want to rollout large PPO batches this can lead to out-of-memory errors when doing the forward passes for text generation and sentiment analysis. We introduce the parameter `forward_batch_size` to split the forward passes into smaller batches. Although this hurts performance a little this is neglectible compared to the computations of the backward passes when optimizing the model. The same parameter is used in the `PPOTrainer` when doing forward passes. The `batch_size` should multiple of `forward_batch_size`.

**Training time**

This step takes **~2h** on a P6000 GPU with the above specified settings.
"""

ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **config)
fbs = config['forward_batch_size']

for epoch in tqdm(range(int(np.ceil(config["steps"]/config['batch_size'])))):
    torch.cuda.empty_cache()
    logs = dict()
    game_data = dict()
    timing = dict()
    t0 = time.time()
    
    #### get a batch from the dataset
    df_batch = df.sample(config['batch_size'])
    game_data['query'] = df_batch['query'].tolist()
    query_tensors = torch.stack(df_batch['tokens'].tolist())
    
    #### get response from gpt2
    t = time.time()
    total_length = config['txt_in_len']+config['txt_out_len']
    response_tensors = []
    for i in range(int(config['batch_size']/fbs)):
        response  = respond_to_batch(gpt2_model, query_tensors[i*fbs:(i+1)*fbs],
                                     txt_len=config['txt_out_len'])
        response_tensors.append(response)
    response_tensors = torch.cat(response_tensors)
    game_data['response'] = [gpt2_tokenizer.decode(response_tensors[i, :]) for i in range(config['batch_size'])]
    timing['time/get_response'] = time.time()-t

    #### tokenize text for sentiment analysis
    t = time.time()
    texts = [q + r for q,r in zip(game_data['query'], game_data['response'])]
    sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
    timing['time/build_input_sentiment'] = time.time()-t

    #### get sentiment score
    t = time.time()
    rewards = []
    for i in range(int(config['batch_size']/fbs)):
        res = sentiment_model.forward(sentiment_inputs[i*fbs:(i+1)*fbs],
                                      attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
        rewards.append(res)
    rewards = torch.cat(rewards)
    timing['time/get_sentiment_preds'] = time.time()-t

    #### Run PPO training 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/optimization'] = time.time()-t
     
    #### Log everything
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(game_data['query'], game_data['response'], rewards.cpu().tolist())]
    logs.update({'game_log':wandb.Table(
        columns=['query', 'response', 'reward'],
        rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
    logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
    logs['env/reward_dist'] = rewards.cpu().numpy()
    wandb.log(logs)

"""### Training progress
If you are tracking the training progress with Weights&Biases you should see a plot similar to the one below. Check out the interactive sample report on wandb.ai: [link](https://app.wandb.ai/lvwerra/trl-showcase/runs/1jtvxb1m/).

<div style="text-align: center">
<img src='images/gpt2_tuning_progress.png' width='800'>
<p style="text-align: center;"> <b>Figure:</b> Reward mean and distribution evolution during training. </p>
</div>

One can observe how the model starts to generate more positive outputs after a few optimisation steps.

> Note: Investigating the KL-divergence will probably show that at this point the model has not converged to the target KL-divergence, yet. To get there would require longer training or starting with a higher inital coefficient.

## Model inspection
Let's inspect some examples from the IMDB dataset. We can use `gpt2_model_ref` to compare the tuned model `gpt2_model` against the model before optimisation.
"""

#### get a batch from the dataset
bs = 16
game_data = dict()
df_batch = df.sample(bs)
game_data['query'] = df_batch['query'].tolist()
query_tensors = torch.stack(df_batch['tokens'].tolist())

#### get response from gpt2 and gpt2_ref
total_length = config['txt_in_len']+config['txt_out_len']
response_tensors_ref  = respond_to_batch(gpt2_model_ref, query_tensors, txt_len=config['txt_out_len'])
game_data['response (before)'] = [gpt2_tokenizer.decode(response_tensors_ref[i, :]) for i in range(bs)]

response_tensors  = respond_to_batch(gpt2_model, query_tensors, txt_len=config['txt_out_len'])
game_data['response (after)'] = [gpt2_tokenizer.decode(response_tensors[i, :]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q,r in zip(game_data['query'], game_data['response (before)'])]
sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
rewards = sentiment_model.forward(sentiment_inputs, attention_masks)[0][:, 1].detach()
game_data['rewards (before)'] = rewards.cpu().numpy()

texts = [q + r for q,r in zip(game_data['query'], game_data['response (after)'])]
sentiment_inputs, attention_masks = build_bert_batch_from_txt(texts, sentiment_tokenizer, device)    
rewards = sentiment_model.forward(sentiment_inputs, attention_masks)[0][:, 1].detach()
game_data['rewards (after)'] = rewards.cpu().numpy()

# store results in a dataframe
df_results = pd.DataFrame(game_data)
df_results

"""Looking at the reward mean/median of the generated sequences we observe a significant difference."""

print('mean:')
display(df_results.mean())
print()
print('median:')
display(df_results.median())

"""## Save model
Finally, we save the model to disk for later usage.
"""

os.makedirs('gpt2-imdb-pos')
gpt2_model.save_pretrained('gpt2-imdb-pos')
gpt2_tokenizer.save_pretrained('gpt2-imdb-pos')

