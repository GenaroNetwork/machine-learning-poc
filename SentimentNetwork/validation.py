from EncryptedSentimentNetwork import *
from UnencryptedSentimentNetwork import *

g = open('reviews.txt', 'r') # What we know!
reviews = list(map(lambda x: x[:-1], g.readlines()))
g.close()

g = open('labels.txt', 'r') # What we WANT to know!
labels = list(map(lambda x: x[:-1].upper(), g.readlines()))
g.close()

# mlp = UnencryptedSentimentNetwork(reviews[:20000], labels[:20000], min_count=10, polarity_cutoff=0.1, hidden_nodes=8, learning_rate=0.01)
# mlp.train(reviews[:20000], labels[:20000])
# mlp.test(reviews[-9000:], labels[-9000:])

# mlp = EncryptedSentimentNetwork(reviews[:10000], labels[:10000], min_count=10, polarity_cutoff=0.1, hidden_nodes=8, learning_rate=0.01)
# mlp.train(reviews[:10000], labels[:10000])
# mlp.test(reviews[-10000:], labels[-10000:])

mlp = UnencryptedSentimentNetwork(reviews[:100], labels[:100], min_count=10, polarity_cutoff=0.1, hidden_nodes=8, learning_rate=0.01)
mlp.train(reviews[:100], labels[:100])
mlp.test(reviews[-9000:], labels[-9000:])

mlp = EncryptedSentimentNetwork(reviews[:100], labels[:100], min_count=10, polarity_cutoff=0.1, hidden_nodes=8, learning_rate=0.01)
mlp.train(reviews[:100], labels[:100])
mlp.test(reviews[-9000:], labels[-9000:])
