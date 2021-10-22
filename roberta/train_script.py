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
        "num_train_epochs": {"values": [5]},
        "learning_rate": {"values": [1e-5]},
    },
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

