# Unit 3: Introduction to Machine Learning

## Supervised and Unsupervised Learning

In supervised learning, the model learns from labelled examples: each input comes with the correct output. Classification predicts a category (spam or not spam), while regression predicts a continuous value (house price). In unsupervised learning there are no labels; the model finds structure on its own, for example grouping customers with k-means clustering or reducing dimensions with PCA.

## Overfitting and Underfitting

A model overfits when it learns noise in the training data instead of the underlying pattern. It scores very well on training data but poorly on new data. A model underfits when it is too simple to capture the pattern, so it performs poorly on both.

Ways to reduce overfitting include collecting more data, using a simpler model, regularization (L1 or L2 penalties on large weights), dropout in neural networks, early stopping, and cross-validation to choose hyperparameters. This tension is known as the bias-variance trade-off: simple models have high bias, complex models have high variance.

## Evaluation Metrics

Accuracy is the share of correct predictions, but it is misleading on imbalanced data. If 99% of transactions are legitimate, a model that always predicts "legitimate" is 99% accurate yet useless for catching fraud.

Precision is the share of positive predictions that are correct: TP / (TP + FP). Recall is the share of actual positives the model finds: TP / (TP + FN). The F1 score is the harmonic mean of precision and recall. Choose precision when false positives are costly (flagging legitimate email as spam) and recall when false negatives are costly (missing a disease or an intrusion).

ROC-AUC measures how well the model ranks positives above negatives across all thresholds. An AUC of 0.5 is random guessing and 1.0 is perfect separation.

## Cross-Validation

Evaluating on the same data used for training gives an over-optimistic score. A held-out test set fixes this but wastes data and gives a noisy estimate. K-fold cross-validation splits the training data into k folds, trains on k-1 of them and validates on the remaining one, repeating k times so each fold is used for validation once. The average score is a more reliable estimate. Stratified k-fold keeps the class proportions the same in every fold, which matters for imbalanced classification.

## Retrieval-Augmented Generation

Large language models can produce fluent but incorrect answers, known as hallucinations, especially about information outside their training data. Retrieval-augmented generation (RAG) reduces this by first retrieving relevant passages from a trusted document collection and then giving those passages to the model as context. Documents are split into chunks, each chunk is converted to an embedding vector, and the vectors are stored in a vector database. At question time the question is embedded too, and the nearest chunks by cosine similarity are retrieved and inserted into the prompt.
