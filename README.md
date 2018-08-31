## Aspect Level Sentiment Classification with Deep Memory Network

[TensorFlow](https://www.tensorflow.org/) implementation of [Tang et al.'s EMNLP 2016](https://arxiv.org/abs/1605.08900) work.

### Problem Statement
Given a sentence and an aspect occurring in the sentence, this task aims at inferring the sentiment polarity (e.g. positive, negative, neutral) of the aspect.

### Example
For example, in sentence ''great food but the service was dreadful!'', the sentiment polarity of aspect ''food'' is positive while the polarity of aspect ''service'' is negative.

### Quick Start
Install this [quick GLOVE embeddings loading tool](https://github.com/vzhong/embeddings)

Runs on python3 and tensorflow 1.4.1

Train a model with 3 hops on the [Restaurant](http://alt.qcri.org/semeval2016/task5/) dataset.
```
python main.py --show True
```

### Performance
Achieved accuracy of 72% for Laptop and 79% for Restaurant.

### Acknowledgements
* More than 80% of the code is borrowed from [ganeshjawahar](https://github.com/ganeshjawahar/mem_absa).
* Using this code means you have read and accepted the copyrights set by the dataset providers.

### Author
Tian Tian

### Licence
MIT
