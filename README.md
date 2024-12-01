# Fine-tuning-T5
The code defines two T5 models for reranking tasks:
1. T5Ranker:

This class inherits from torch.nn.Module and is designed for reranking tasks.
It takes a pre-trained T5 model (google-t5/t5-base) and its tokenizer.
It defines methods for:
Saving and loading the model state.
Tokenizing text (excluding special tokens).
Forward pass:
Processes query and document tokens with special prefixes ("query:", "document:", "relevant:")
Concatenates these tokens and attention masks for the T5 model.
Removes padding tokens.
Calls the T5 model with these inputs and returns the loss.
Generate pass:
Similar to the forward pass, but it uses generate method instead of forward for getting scores.
Returns the mean score for each query-document pair.
2. T5_with_reranking_loss:

This class also inherits from torch.nn.Module.
It defines similar functionalities as T5Ranker, but with some differences:
Uses different special prefixes ("query:", "document:").
In the forward pass, it uses decoder input with padding tokens.
It calculates the mean score directly from the first logits instead of generating.
The main function:**
Defines hyperparameters for training and validation.
Sets the random seed for reproducibility.
Defines a dictionary mapping model names to their corresponding class.
Takes arguments for model type, dataset, training data, validation data, and output directory.
Creates the model based on the chosen type.
Defines an optimizer with different learning rates for T5 parameters and other model parameters.
Implements a training loop with the following steps:
Trains the model for a maximum of MAX_EPOCH epochs.
In each epoch:
Calculates the training loss.
Validates the model on the validation data and calculates a score (MRR@100).
Saves the model if it achieves the best validation score so far.
Implements early stopping if the validation score doesn't improve for PATIENCE epochs.
Loads the best model based on the validation score.
Returns the model and the epoch where it achieved the best score.
Additional functions:**
train_iteration: Handles training iterations with gradient accumulation.
validate: Evaluates the model on the validation data and returns the MRR@100 score.
run_model: Generates scores for each query-document pair and returns them in a dictionary.
main_cli: Defines command-line arguments for running the script.
Overall, this code provides a framework for training and validating T5 models for reranking tasks with a custom loss function.Fine tuning T5
