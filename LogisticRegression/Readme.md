# Readme

A program that can judge a mail is spam or ham using logistic regression.

```python
# the parameters can be adjusted
ITERATIONS = 10 # iterations while learning
ALPHA = 0.1     # learning rate
SCALING_FACTOR = 10000  # scaling factor while encryption

# spams: list of spams; hams: list of hams; mail: a new mail to be predicted
model = logisticregression.LogisticRegression(spams, hams, ITERATIONS, ALPHA)
pred = model.predict(mail)

model.encrypt(logisticregression.pubkey, SCALING_FACTOR)
encrypted_pred = model.predict(mail)
```