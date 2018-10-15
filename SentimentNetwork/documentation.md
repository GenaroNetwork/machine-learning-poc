# Readme（参考 https://cloud.tencent.com/developer/article/1043113,https://github.com/udacity/deep-learning/tree/master/sentiment-network,https://blog.csdn.net/weiwei9363/article/category/7233487）

代码大概是在udacity的sentiment-network基础上增加了同态加密，udacity的sentiment-network参见：https://github.com/udacity/deep-learning/blob/master/sentiment-network/Sentiment_Classification_Solutions.ipynb

H_sigmoid_txt[0][0] = 0.5
H_sigmoid_txt[0][1] = 0.25
H_sigmoid_txt[0][2] = -1/48.0
H_sigmoid_txt[0][3] = 1/480.0
H_sigmoid_txt[0][4] = -17/80640.0

sigmoid函数：1 / (1 + e**(-x))用泰勒级数展开如下（**在python中表示次方）：
(1 / 2) + (x / 4) - (x**3 / 48) + (x**5 / 480) - (x**7 * 17 / 80640) + ......

- 问题

classification.py存在两个load_linear_transformation函数，参考网站中只有第二个函数。但是encrypt_login.py中的load_linear_transformation函数和第一个相同。

- 功能

利用BP神经网络判断一段影评是正面评价(Positive)还是负面评价(Negative)

- NeuralNetwork类

两层的神经网络。

    - 参数：
    reviews(list) - List of reviews used for training
    labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
    min_count(int) - Words should only be added to the vocabulary 
                        if they occur more than this many times
    polarity_cutoff(float) - The absolute value of a word's positive-to-negative
                                ratio must be at least this big to be considered.
    hidden_nodes(int) - Number of nodes to create in the hidden layer
    learning_rate(float) - Learning rate to use while training
