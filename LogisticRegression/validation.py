import logisticregression
import sys

print("Importing dataset from disk...")
f = open('spam.txt', 'r')
raw = f.readlines()
f.close()

spam = list()
for row in raw:
    spam.append(row[:-2].split(" "))
    
f = open('ham.txt', 'r')
raw = f.readlines()
f.close()

ham = list()
for row in raw:
    ham.append(row[:-2].split(" "))

train_spam_begin = 0
train_spam_end = 1000
train_ham_begin = 0
train_ham_end = 1000

predict_spam_begin = -1000
predict_spam_end = len(spam) - 0
predict_ham_begin = -1000
predict_ham_end = len(ham) - 0

ITERATIONS = 10
ALPHA = 0.1     # learning rate
SCALING_FACTOR = 10000

model = logisticregression.LogisticRegression(spam[train_spam_begin:train_spam_end], ham[train_ham_begin:train_ham_end], ITERATIONS, ALPHA)

unencrypted_fp = 0
unencrypted_tn = 0
unencrypted_tp = 0
unencrypted_fn = 0

for i, h in enumerate(ham[predict_ham_begin:predict_ham_end]):
    pred = model.predict(h)

    if(pred < 0.5):
        unencrypted_tn += 1
    else:
        unencrypted_fp += 1

for i, h in enumerate(spam[predict_spam_begin:predict_spam_end]):
    pred = model.predict(h)

    if(pred > 0.5):
        unencrypted_tp += 1
    else:
        unencrypted_fn += 1

print("Unencrypted Accuracy: %" + str(100 * (unencrypted_tn + unencrypted_tp) / float(unencrypted_tn +
      unencrypted_tp + unencrypted_fn + unencrypted_fp))[0:6])
print("False Positives: %" + str(100 * unencrypted_fp / float(unencrypted_tp + unencrypted_fp))[0:6] +
      "  <- privacy violation level")
print("False Negatives: %" + str(100 * unencrypted_fn / float(unencrypted_tn + unencrypted_fn))[0:6] +
      "  <- security risk level")

model.encrypt(logisticregression.pubkey, SCALING_FACTOR)

# generate encrypted predictions. Then decrypt them and evaluate.

encrypted_fp = 0
encrypted_tn = 0
encrypted_tp = 0
encrypted_fn = 0

for i, h in enumerate(ham[predict_ham_begin:predict_ham_end]):
    encrypted_pred = model.predict(h)
    try:
        pred = logisticregression.prikey.decrypt(encrypted_pred) / model.scaling_factor

        if(pred < 0):
            encrypted_tn += 1
        else:
            encrypted_fp += 1
    except:
        print("overflow")

    if i % 10 == 0:
        sys.stdout.write('\r I:'+str(encrypted_tn + encrypted_tp + encrypted_fn + encrypted_fp) +
                         "  Correct: %" + str(100 * encrypted_tn / float(encrypted_tn + encrypted_fp))[0:6])

for i, h in enumerate(spam[predict_spam_begin:predict_spam_end]):
    encrypted_pred = model.predict(h)
    try:
        pred = logisticregression.prikey.decrypt(encrypted_pred) / model.scaling_factor

        if pred > 0:
            encrypted_tp += 1
        else:
            encrypted_fn += 1
    except:
        print("overflow")

    if i % 10 == 0:
        sys.stdout.write('\r I:'+str(encrypted_tn + encrypted_tp + encrypted_fn + encrypted_fp) + "  Correct: %" +
            str(100 * (encrypted_tn + encrypted_tp) / float(encrypted_tn + encrypted_tp + encrypted_fn + encrypted_fp))[0:6])

sys.stdout.write('\r I:'+str(encrypted_tn + encrypted_tp + encrypted_fn + encrypted_fp) + "  Correct: %" +
    str(100 * (encrypted_tn + encrypted_tp) / float(encrypted_tn + encrypted_tp + encrypted_fn + encrypted_fp))[0:6])

print("Encrypted Accuracy: %" + str(100 * (encrypted_tn + encrypted_tp) / float(encrypted_tn + encrypted_tp + encrypted_fn + encrypted_fp))[0:6])
print("False Positives: %" + str(100 * encrypted_fp / float(encrypted_tp + encrypted_fp))[0:6] + "  <- privacy violation level")
print("False Negatives: %" + str(100 * encrypted_fn / float(encrypted_tn + encrypted_fn))[0:6] + "  <- security risk level")
