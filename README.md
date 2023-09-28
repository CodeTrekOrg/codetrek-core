# codetrek-core

Command line arguments that CodeTrek provides along with their default values are as follows:
```
data_dir='tmp/data',
output_dir='tmp',
model_dir='tmp/model',
split='80:10',
phase='train',
device=-1
```

For training and testing, you will need to provide the following hyperparameters:
```
seed,
embed_dim,
nhead,
dropout,
grad_clip,
transformer_layers,
dim_feedforward,
num_epochs,
batch_size,
learning_rate,
iter_per_epoch,
num_walks,
num_steps,
top_k_walks
```