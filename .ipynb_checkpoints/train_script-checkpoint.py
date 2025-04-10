from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
import matplotlib.pyplot as plt
from transformers import EarlyStoppingCallback, get_cosine_with_hard_restarts_schedule_with_warmup
import json
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch
from torch.optim import AdamW
from sentence_transformers import InputExample
from transformers import TrainerCallback
from datasets import load_dataset
from sentence_transformers.sampler import NoDuplicatesBatchSampler

model = SentenceTransformer("mixedbread-ai/deepset-mxbai-embed-de-large-v1", device="cuda")

model.max_seq_length = 200

# Lade den Datensatz im Streaming-Modus
train_dataset = load_from_disk("Data/train_dataset")
eval_dataset = load_from_disk("Data/dev_dataset")
test_dataset = load_from_disk("Data/test_dataset")

filtered_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in ['anchor', 'positive', 'hard_negative_1', 'hard_negative_2']])
#print(f"Length of train dataset {len(train_dataset['train'])}")
train_dataset = filtered_dataset

model_path = "Training/finetuning_v1"

# Subsampling
#train_dataset = train_dataset.select(range(2000))
#eval_dataset = eval_dataset.select(range(500))
#test_dataset = test_dataset.select(range(500))

#Subsampling
print(f"Trainingsdatensatz: {train_dataset}")
print(f"Evaluationsdatensatz: {eval_dataset}")
print(f"Testdatensatz: {test_dataset}")

loss = MultipleNegativesRankingLoss(model)

# For Cosine-Accuracy before training
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["anchor"],
    positives=eval_dataset["positive"],
    negatives=eval_dataset["hard_negative"],
    name="mxbai-dev",
)

args = SentenceTransformerTrainingArguments(
    output_dir=f"{model_path}",
    num_train_epochs=15,
    lr_scheduler_type="constant",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=128,
    learning_rate=1e-5,
    #warmup_ratio=0.0,
    #dataloader_drop_last=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=200,
    #eval_steps=1000,
    #save_steps=3000,
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
    fp16=True,
    bf16=False,
    #data_seed=10,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    metric_for_best_model="mxbai-dev_cosine_accuracy",
    load_best_model_at_end=True,
    weight_decay=0.001,
    max_grad_norm=1.0,
    eval_on_start=True,
)

# Optimizer
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

total_steps = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
T_0 = total_steps // args.num_train_epochs

# Optimierter Scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=T_0,
    T_mult=2,
    eta_min=1e-6
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    evaluator=dev_evaluator,
    loss=loss,
    #optimizers=(optimizer, scheduler),
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.05),TrainerCallback()],
)

#dev_result = dev_evaluator(model)
#dev_score = dev_result['mxbai-dev_cosine_accuracy']
dev_score=0.5100276470184326
#print(f"Dev Accuracy: {dev_score}")

print("Start Training")
trainer.train()

# # For Cosine-Accuracy after training
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["hard_negative"],
    name="mxbai-test",
)
test_result = test_evaluator(model)
test_score = test_result['mxbai-test_cosine_accuracy']

log_history = trainer.state.log_history
log_file_path = f"{model_path}/train_logs.txt"

# Logs in eine Datei schreiben
with open(log_file_path, "w") as log_file:
    for log_entry in log_history:
        log_file.write(json.dumps(log_entry) + "\n")

print(f"Logs wurden in {log_file_path} gespeichert.")

train_steps, train_loss, lr_values = [], [], []
eval_steps, eval_loss = [], []
cosine_accuracies = []

# Extrahieren der relevanten Werte aus den Logs
for entry in log_history:
    if 'loss' in entry and 'learning_rate' in entry:
        train_steps.append(entry['step'])
        train_loss.append(entry['loss'])
        lr_values.append(entry['learning_rate'])
    elif 'eval_loss' in entry:
        eval_steps.append(entry['step'])
        eval_loss.append(entry['eval_loss'])
    elif 'eval_mxbai-dev_cosine_accuracy' in entry:
        eval_steps.append(entry['step'])
        cosine_accuracies.append(entry['eval_mxbai-dev_cosine_accuracy'])

# Plot: Train Loss, Cosine Accuracy und Learning Rate
plt.figure(figsize=(12, 6))

# Achse 1: Train Loss
ax1 = plt.gca()
ax1.plot(train_steps, train_loss, 'b-', label='Train Loss')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Achse 2: Cosine Accuracy
ax2 = ax1.twinx()
ax2.plot(eval_steps, cosine_accuracies, 'g-', marker='o', label='Cosine Accuracy')
ax2.set_ylabel('Cosine Accuracy', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Achse 3: Learning Rate
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))  # Zweite rechte Achse etwas nach außen verschieben
ax3.plot(train_steps, lr_values, 'r--', label='Learning Rate')
ax3.set_ylabel('Learning Rate', color='g')
ax3.tick_params(axis='y', labelcolor='g')

# Titel und Legenden
plt.title('Loss, Cosine Accuracy and Learning Rate during Training')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()
ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')

plt.tight_layout()
plt.savefig(f"{model_path}/loss_cosine_accuracy_lr.png")
plt.close()

# Modell speichern
model.save_pretrained(f"{model_path}/model_finetuned")
