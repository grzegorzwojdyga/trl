from sklearn.model_selection import train_test_split
import pandas as pd

from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)

import logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

from sklearn.model_selection import train_test_split
import wandb


train_df = pd.read_csv('EnoughTrain.csv')
print(train_df.head())

X_train, X_test = train_test_split(train_df, test_size=0.1, random_state=43)
X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=43) # 0.25 x 0.9 = 0.2

sweep_config = {
    "method": "grid",  # grid, random
    "metric": {"name": "eval_loss", "goal": "minimize"},
       "parameters": {
        "num_train_epochs": {"values": [2, 3, 5]},
        "learning_rate": {"values": [5e-4, 1e-5, 5e-5, 1e-6]},
    },
}

old_model_args = {
    'fp16':False,
 #   'wandb_project': 'Roberta Colab',
 #   'num_train_epochs': 3,
    'overwrite_output_dir':True,
    #'learning_rate': 1e-5,
    #'use_early_stopping': True,
    #'early_stopping_delta': 0.01,
    #'early_stopping_metric': 'mcc',
    #'early_stopping_metric_minimize': False,
    #'early_stopping_patience': 5,
    'evaluate_during_training': True,
    #'evaluate_during_training_steps': 1000,
    'reprocess_input_data': True,
    'manual_seed': 4,
    'use_multiprocessing': True,
    'train_batch_size': 16,
    'eval_batch_size': 8,
  #  'labels_list': ["true", "false"]
}

model_args = ClassificationArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.evaluate_during_training = True
model_args.manual_seed = 4
model_args.use_multiprocessing = True
model_args.train_batch_size = 16
model_args.eval_batch_size = 8
model_args.wandb_project = "Eagle Roberta Sweep"

sweep_id = wandb.sweep(sweep_config, project="Roberta Eagle")

def train():
    # Initialize a new wandb run
    wandb.init()

    # Create a TransformerModel
    model = ClassificationModel(
        "roberta",
        "roberta-base",
        use_cuda=True,
        args=model_args,
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(X_train, eval_df=X_val)

    # Evaluate the model
    model.eval_model(X_test)

    # Sync wandb
    wandb.join()

wandb.agent(sweep_id, train)

