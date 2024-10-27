import os
import os.path as op
import time
import logging

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import torchmetrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from watermark import watermark

# Importing custom dataset utilities for data loading, processing, and partitioning
from local_dataset_utilities import (
    download_dataset,
    load_dataset_into_to_dataframe,
    partition_dataset,
    IMDBDataset,
)

# Setting up logging to capture detailed information about the process
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to tokenize text data using the specified tokenizer


def tokenize_text(batch):
    # Tokenizes text data with truncation and padding enabled
    return tokenizer(batch["text"], truncation=True, padding=True)

# Function to perform training and validation


def train(num_epochs, model, optimizer, train_loader, val_loader, device):
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        # Initialize accuracy metric for the training process
        train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=2).to(device)

        # Loop through each batch in the training data
        for batch_idx, batch in enumerate(train_loader):
            model.train()  # Set model to training mode

            # Move input data and labels to the specified device (CPU or GPU)
            for s in ["input_ids", "attention_mask", "label"]:
                batch[s] = batch[s].to(device)

            # Forward Pass and Backward Propagation
            # Perform a forward pass through the model to compute loss
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )

            # Reset gradients before backward pass to prevent accumulation
            optimizer.zero_grad()
            # Perform backward propagation to compute gradients
            outputs["loss"].backward()

            # Update Model Parameters
            # Apply optimizer step to update model weights
            optimizer.step()

            # Logging Training Progress
            # Log progress every 300 batches
            if not batch_idx % 300:
                logger.info(
                    f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(train_loader):04d} | Loss: {outputs['loss']:.4f}"
                )

            # Switch to evaluation mode and calculate training accuracy
            model.eval()
            with torch.no_grad():
                # Get predicted labels and update training accuracy metric
                predicted_labels = torch.argmax(outputs["logits"], dim=1)
                train_acc.update(predicted_labels, batch["label"])

        # Validation Phase
        with torch.no_grad():
            model.eval()  # Set model to evaluation mode
            val_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=2).to(device)

            # Loop through each batch in the validation data
            for batch in val_loader:
                # Move input data and labels to the specified device (CPU or GPU)
                for s in ["input_ids", "attention_mask", "label"]:
                    batch[s] = batch[s].to(device)

                # Perform a forward pass through the model to compute outputs
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["label"],
                )

                # Get predicted labels and update validation accuracy metric
                predicted_labels = torch.argmax(outputs["logits"], dim=1)
                val_acc.update(predicted_labels, batch["label"])

            # Log accuracy for the current epoch
            logger.info(
                f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Train acc.: {train_acc.compute()*100:.2f}% | Val acc.: {val_acc.compute()*100:.2f}%"
            )


# Main script entry point
if __name__ == "__main__":
    # Print system and package information
    print(watermark(packages="torch,transformers", python=True))
    logger.info("Torch CUDA available? %s", torch.cuda.is_available())

    # Define device (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Set random seed for reproducibility
    torch.manual_seed(123)

    ##########################
    # 1. Loading the Dataset
    ##########################

    # Download dataset if not present locally
    download_dataset()
    # Load dataset into a DataFrame for easy manipulation
    df = load_dataset_into_to_dataframe()

    # Partition dataset into train, validation, and test splits if not already done
    if not (op.exists("train.csv") and op.exists("val.csv") and op.exists("test.csv")):
        partition_dataset(df)

    # Load CSV files as Hugging Face datasets
    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "val.csv",
            "test": "test.csv",
        },
    )

    #########################################
    # 2. Tokenization and Numericalization
    #########################################

    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    logger.info("Tokenizer input max length: %d", tokenizer.model_max_length)
    logger.info("Tokenizer vocabulary size: %d", tokenizer.vocab_size)

    # Apply tokenization to the entire dataset
    logger.info("Tokenizing dataset...")
    imdb_tokenized = imdb_dataset.map(
        tokenize_text, batched=True, batch_size=None)
    del imdb_dataset  # Free memory by deleting original dataset

    # Set dataset format to PyTorch tensors for DataLoader compatibility
    imdb_tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    #########################################
    # 3. Set Up DataLoaders
    #########################################

    # Create custom dataset partitions for train, validation, and test
    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    # Create DataLoader objects for batching and loading data
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=12, shuffle=True, num_workers=1, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=12, num_workers=1, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=12, num_workers=1, drop_last=True)

    #########################################
    # 4. Initializing the Model
    #########################################

    # Load a pre-trained DistilBERT model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
    model.to(device)  # Move model to specified device (GPU or CPU)

    # Set up an optimizer for training
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    #########################################
    # 5. Finetuning the Model
    #########################################

    # Record start time for training duration calculation
    start = time.time()

    # Train the model over specified epochs
    train(num_epochs=3, model=model, optimizer=optimizer,
          train_loader=train_loader, val_loader=val_loader, device=device)

    # Calculate and log training time
    end = time.time()
    elapsed = end - start
    logger.info("Time elapsed %.2f min", elapsed / 60)

    #########################################
    # 6. Testing the Model
    #########################################

    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=2).to(device)

        # Loop through test batches for evaluation
        for batch in test_loader:
            for s in ["input_ids", "attention_mask", "label"]:
                batch[s] = batch[s].to(device)

            # Perform a forward pass through the model
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )

            # Calculate predicted labels and update test accuracy metric
            predicted_labels = torch.argmax(outputs["logits"], dim=1)
            test_acc.update(predicted_labels, batch["label"])

    # Log test accuracy result
    logger.info("Test accuracy: %.2f%%", test_acc.compute() * 100)
