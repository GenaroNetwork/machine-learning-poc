# doc

- TP/TN/FP/FN

  - TP -- True Positive （真正, TP）被模型预测为正的正样本；可以称作判断为真的正确率
  - TN -- True Negative（真负 , TN）被模型预测为负的负样本 ；可以称作判断为假的正确率
  - FP -- False Positive （假正, FP）被模型预测为正的负样本；可以称作误报率
  - FN -- False Negative（假负 , FN）被模型预测为负的正样本；可以称作漏报率
  - TPR -- True Positive Rate（真正率 , TPR）或灵敏度（sensitivity）
　　TPR = TP /（TP + FN）
　　被预测为正的正样本结果数 / 正样本实际数
  - TNR -- True Negative Rate（真负率 , TNR）或特指度（specificity）
　　TNR = TN /（TN + FP）
　　被预测为负的负样本结果数 / 负样本实际数
  - FPR -- False Positive Rate （假正率, FPR）
　　FPR = FP /（TN + FP）
　　被预测为正的负样本结果数 /负样本实际数
  - FNR -- False Negative Rate（假负率 , FNR）
　　FNR = FN /（TP + FN）
　　被预测为负的正样本结果数 / 正样本实际数

- spam：垃圾邮件

- ham：正常邮件

- logisticregression类

  - logistic回归，此处用于判断邮件是否为垃圾邮件。由于事先知道每个样本是否为垃圾邮件，如果为垃圾邮件，则定义该邮件的target等于1，否则target等于0。对每封邮件计算一个预期值pred，该值范围为(0, 1)，如果大于0.5定义为垃圾邮件，且值越接近1，越像垃圾邮件。iterations的值代表样本学习的次数。
  - 同时用到了半同态加密（完全同态加密算法运算速度比较慢），即利用加密后的权重weight进行相似运算，即未经半同态加密的权重和经过半同态加密的权重都可以判断邮件是否为垃圾邮件，但是结果可能稍有不同。

- 算法
  将一定数量的邮件（对应ham.txt或spam.txt的一行）作为样本传递给LogisticRegression类。该类分别对每封邮件进行训练（iterations设置每个样本训练的次数，alpha设置学习速率，建议保证iterations乘以alpha的值不能太小（例如可以为1，5等值），并且alpha不能太大也不能太小（例如可以为0.1，0.01等值）），训练过程如下1, 2步骤：
  1. 首先对每封邮件调用predict方法（该方法根据是否使用半同态加密，分别调用encrypted_predict和unencrypted_predict方法，在训练的时候由于尚未进行半同态加密，调用的是unencrypted_predict方法），该方法算法如下：将邮件拆分成一个个单词，每个单词都有一个随机的初始权重weight（限定在[-0.05, 0.05)之间），将所有单词的权重相加，再调用sigmoid（S型曲线）将相加的结果映射到(0, 1)的空间中，该值即为pred。
  PS：将pred映射到(0, 1)的空间的原因是当邮件为垃圾邮件时，target值为1，正常邮件的target值为0，而pred需要通过训练不断趋近target。
  2. delta = target - pred就代表着样本的预期结果和实际结果的差距，对该邮件的每个单词的weight加上(target - pred) * alpha，即根据预期结果和实际结果的差距调节每个单词的权重。
  3. 训练完毕后，对某封邮件进行是否为垃圾邮件的判定时，调用predict方法得到相应pred，如果pred大于0.5，则认定为垃圾邮件。

  PS：
  - unencrypted_predict方法中将pred映射到(0, 1)的空间，而encrypted_predict方法并没有这个操作，原因是半同态加密不支持sigmoid方法中的某些操作，例如次方，由于encrypted_predict方法并没有在训练样本时调用，不会计算其pred和target的差值，该方法只用在判定未知邮件是否为垃圾邮件时用到，所以其计算出来的pred范围在整个实数空间内，需要将该pred和0（而不是0.5，大于0的数经过sigmoid映射后范围为(0.5, 1)）进行比较，即大于0判定为垃圾邮件。
  - 对weight进行半同态加密时进行了如下操作：int(min(weight,self.maxweight) * self.scaling_factor)，其中，调用int函数以及乘以self.scaling_factor的原因是用python的phe库进行浮点数的加法运算时容易出现OverFlowError（例如prikey.decrypt(pubkey.encrypt(989.1) + pubkey.encrypt(0.1))有时会报OverFlowError的错），至于为何要调用min函数将weight限制为maxweight（10）以内，可能是防止weight相加后数值太大导致溢出。但是进行min操作后可能会导致半同态加密的运算结果和未经半同态加密的运算结果有微小的差距。
  - 关于encrypt函数的第二个参数scaling_factor：该参数的作用是在用半同态加密进行运算时将weight转换为整型，从而防止半同态加密操作浮点数容易出现的OverFlowError，其值不能太小，太小的话丢失了weight的精度，也不能太大，太大的话多个weight相加后数值太大也可能导致溢出。
  - 建议保证iterations乘以alpha的值不能太小（例如可以为1，5等值），并且alpha不能太大也不能太小的原因：如果iterations乘以alpha的值太小，那么单词经过iterations次学习，每次的学习速率为alpha，可能导致最终的weight调整不到位（学习iterations次的原因是让weight调整到经过predict函数计算出的pred和target的值差距尽量小，即learn函数的返回值尽量小，对于整个email来说就是train函数中打印的Loss值尽可能小）；如果iterations乘以alpha的值太大，那么迭代次数过多浪费时间。如果学习速率alpha太大，那么weight的调整精度太粗，难以到达较合适大值；如果学习速率alpha太小，那么为了保证iterations乘以alpha的值合适，iterations大值将会太大，从而学习时间太长。
  - 关于LogisticRegression构造函数的如下代码：
    self.weights = (np.random.rand(len(vocab)) - 0.5) * 0.1
  为什么要乘以0.1？我的理解是，这个值设置为多少没有太大意义，只是这里将weigth规范到（-0.05, 0.05）之间，当然也可以不乘以0.1，或乘以0.01，不同的weight范围可能会导致不同的准确度，不过要注意scaling_factor和weight的关系，以及maxweight的值和weight也有关。

- 可调参数：
  - iterations（LogisticRegression构造函数的倒数第二个参数，用于调整train函数中的迭代次数，合适的值可以提高准确率）
  - alpha（LogisticRegression构造函数的倒数第一个参数，学习速率，合适的值可以提高准确率）
  - scaling_factor（LogisticRegression的encrypt函数的第二个参数，半同态加密中weight的放大系数，合适的值可以尽量保证半同态加密不丢失精度）
