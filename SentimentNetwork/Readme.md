# Readme

A program that can judge a comment on a film is positive or negative using neural network.

```python
# parameters of the neural network:
# 1. min_count(int) - Words should only be added to the vocabulary if they occur more than this many times
# 2. polarity_cutoff(float) - The absolute value of a word's positive-to-negative ratio must be at least this big to be considered.
# 3. hidden_nodes(int) - Number of nodes to create in the hidden layer
# 4. learning_rate(float) - Learning rate to use while training

mlp = UnencryptedSentimentNetwork(training_reviews, training_labels, min_count=10, polarity_cutoff=0.1, hidden_nodes=8, learning_rate=0.01)
mlp.train(training_reviews, training_labels)
mlp.test(new_reviews, new_labels)
```
