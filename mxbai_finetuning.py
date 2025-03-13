from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import pickle
from datasets import load_from_disk
import random
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from datasets import Dataset

# 1. Load a model to finetune with
model = SentenceTransformer("Data/models/mxbai/mxbai-model_v1", device="cuda")

model.max_seq_length = 200

train_dataset = load_from_disk("Data/models/mxbai/train_dataset")
eval_dataset = load_from_disk("Data/models/mxbai/dev_dataset")
test_dataset = load_from_disk("Data/models/mxbai/test_dataset")

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)

# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="Data/models/mxbai",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    dataloader_drop_last=True,
    lr_scheduler_type='cosine',
    eval_strategy="epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="mxbai-test",
)
test_evaluator(model)

