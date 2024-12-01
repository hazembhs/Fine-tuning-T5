## T5 Models for Reranking Tasks

This repository provides two T5-based models for reranking tasks: `T5Ranker` and `T5_with_reranking_loss`. Below is a detailed description of their functionality.

---

### 1. T5Ranker

This class is designed for reranking tasks and inherits from `torch.nn.Module`.

#### Features:
- Utilizes a pre-trained T5 model (`google-t5/t5-base`) and its tokenizer.
- Provides methods for:
  - **Saving and loading** the model state.
  - **Tokenizing text**, excluding special tokens.

#### Core Methods:
- **Forward Pass**:
  - Processes query and document tokens with special prefixes (`"query:"`, `"document:"`, `"relevant:"`).
  - Concatenates these tokens and attention masks for the T5 model.
  - Removes padding tokens.
  - Calls the T5 model with these inputs and returns the loss.

- **Generate Pass**:
  - Similar to the forward pass but uses the `generate` method instead of `forward` for getting scores.
  - Returns the mean score for each query-document pair.

---

### 2. T5_with_reranking_loss

This class also inherits from `torch.nn.Module` and provides reranking capabilities similar to `T5Ranker`.

#### Key Differences:
- Uses different special prefixes (`"query:"`, `"document:"`).
- In the **forward pass**:
  - Includes decoder input with padding tokens.
  - Calculates the mean score directly from the first logits instead of generating scores.

---

### Main Function

The main training and validation process is defined in the `main` function.

#### Features:
- **Hyperparameter Definitions**:
  - Includes settings for training and validation, such as `MAX_EPOCH`, `PATIENCE`, learning rates, and batch sizes.
  
- **Random Seed**:
  - Ensures reproducibility by setting a random seed for all components.

- **Model Initialization**:
  - Maps model names to their corresponding class using a dictionary.
  - Creates the model instance based on the chosen type.

- **Training Loop**:
  - Trains the model for a maximum of `MAX_EPOCH` epochs.
  - Each epoch involves:
    1. Calculating the training loss.
    2. Validating the model on the validation data and computing the MRR@100 score.
    3. Saving the model if it achieves the best validation score.
    4. Implementing early stopping if the validation score does not improve for `PATIENCE` epochs.
  - Loads the best model based on the validation score at the end of training.

- **Returns**:
  - The model and the epoch where it achieved the best validation score.

---

### Additional Functions

1. **`train_iteration`**:
   - Handles training iterations with gradient accumulation.

2. **`validate`**:
   - Evaluates the model on the validation data.
   - Returns the **MRR@100** score.

3. **`run_model`**:
   - Generates scores for each query-document pair.
   - Returns these scores in a dictionary.

4. **`main_cli`**:
   - Provides command-line arguments for running the script.
   - Allows customization of model type, dataset, training data, validation data, and output directory.

---

### Summary

This framework provides robust tools for training and validating T5 models in reranking tasks. It includes support for custom loss functions, hyperparameter tuning, and performance evaluation using the MRR@100 metric.

---

### Acknowledgments

This implementation leverages the Hugging Face `transformers` library and builds on advancements in T5-based architectures for information retrieval. Special thanks to the open-source community for their contributions.
