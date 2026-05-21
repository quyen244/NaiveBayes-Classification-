import re 
import pickle
from collections import defaultdict, Counter
from typing import List 
import numpy as np 
from math import prod
from sklearn.metrics import f1_score , accuracy_score, precision_score , recall_score

X = [
    # spam
    'claim your free iphone now',
    'limited time promotion buy now',
    'win cash reward click this link',
    'cheap crypto investment guaranteed profit',
    'urgent your account has been compromised',
    'exclusive discount for selected users',
    'earn money fast from home',
    'free voucher waiting for you',
    'this is a spam email',
    'buy this product with 50 percent promotion',

    # non spam
    'hi how are you doing today',
    'let us schedule a meeting tomorrow',
    'i am looking forward to seeing you',
    'please review the attached document',
    'can we have lunch next week',
    'thank you for your support',
    'the project deadline is next friday',
    'i will call you later tonight',
    'happy birthday wish you all the best',
    'the weather is very nice today'
]

# 1 = spam, 0 = non-spam
y = [
    1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0
]

X_test = [
    'free cash promotion',
    'schedule a project meeting',
    'earn fast money now',
    'happy to see you tomorrow'
]

y_test = [1,0,1,0]

# 1 spam  , 0 non-spam 

class NaiveBayes:
  def __init__(self, samples : List[str], y : List[int] , vocab_path : str = None):

    self.y = np.array(y)
    self.X = np.array(samples) 
    self.M = len(samples)

    if vocab_path:
        self.vocabulary = pickle.load(vocab_path)
    self.vocabulary = self.__build_vocabulary(samples)

    self.labels , self.prior = self.___build_prior()

    self.likelihood = self.___build_likelihood()

  def predict(self , X_test : List[str]):
    predictions = []
    for sample in X_test:
      encoded_sample = self.__representation(sample)

      probs = []

      for label in self.labels:
        score = 0 
        score += np.log(self.prior[label])
        score += np.sum(np.log([self.likelihood[label][idx] for idx in encoded_sample]))
        probs.append(round(score , 5))
  
      predictions.append(np.argmax(probs))
    
    return predictions

  def save(self, path : str):
    # path_vocab = f'{path}/vocabulary.json'
    # path_likelihood = f'{path}/likelihood.json'
    # path_prior = f'{path}/prior.json'
    # path_label = f'{path}/label.json'

    # pickle.dump(self.vocabulary , path_vocab)
    # pickle.dump(self.likelihood , path_likelihood)
    # pickle.dump(self.prior , path_prior)
    # pickle.dump(self.labels , path_label)
    pass


  def __build_vocabulary(self , samples):
    samples = [self.__preprocess(sample) for sample in samples]

    vocabulary = {
        'unk' : -1
    }

    sorted_words = sorted(
        Counter(" ".join(samples).split()).items(),
        key=lambda x: x[1] , reverse=True
      )
    
    MAX_VOCAB = 10000

    sorted_words = [word for word , freq in sorted_words][:MAX_VOCAB]

    vocabulary.update({word: idx for idx, word in enumerate(sorted_words)})

    return vocabulary

  def ___build_prior(self):
    labels = sorted(np.unique(self.y))

    prior = [round(sum(self.y == label) / self.M , 3) for label in labels]

    return labels , prior

  def  ___build_likelihood(self):
    likelihood = {label : {} for label in self.labels }

    filter = {label : self.X[(self.y == label)] for label in self.labels}

    total_vocab = len(self.vocabulary)

    count_w = {label : Counter(" ".join(filter[label]).split()) for label in self.labels}

    N_c = {label : sum((Counter(" ".join(filter[label]).split()).values())) for label in self.labels}
    
    for word , idx in self.vocabulary.items():
      for label in self.labels:
        # np.char.find(filter[label], word : Nó trả về vị trí xuất hiện đầu tiên của substring trong mỗi string.
        # 
        likelihood[label][idx] = round( (count_w[label][word] + 1) / (N_c[label] + total_vocab) , 3)

        # print(f'word {word} belongs to class {label} with likelihood = {likelihood[label][idx] }')

    # for label in self.labels:
    #     total_likelihood = sum(likelihood[label].values())
    #     print(f'class {label} with {total_likelihood}')

    return likelihood

  def __representation(self , sample : str):
    sample = sample.split()
    return [self.vocabulary.get(word , -1) for word in sample] 

  def __preprocess(self, sample : str):
    sample = sample.lower()
    return sample 

class Evaluator:
  @staticmethod
  def evaluate(ground_truth : List[int] , predictions : List[int]):
    report = {}

    report['macro_f1_score'] = f1_score(y_true=ground_truth , y_pred = predictions , average='macro')
    report['weighted_f1_score'] = f1_score(y_true=ground_truth , y_pred = predictions , average='weighted')
    report['precision'] = precision_score(y_true=ground_truth , y_pred = predictions)
    report['recall'] = recall_score(y_true=ground_truth , y_pred = predictions)

    print(report)

    return report

if __name__ == "__main__":
  

    classifier = NaiveBayes(X , y)
    predictions = classifier.predict(X_test)

    report = Evaluator.evaluate(y_test , predictions)