import pandas as pd
import numpy as np
import re
import nltk
from typing import TypeVar, Callable
from numpy.linalg import norm
dframe = TypeVar('pd.core.frame.DataFrame')

char_set = '#!abcdefghijklmnopqrstuvwxyz'
narray = TypeVar('numpy.ndarray')

nltk.download('stopwords')
from nltk.corpus import stopwords
swords = stopwords.words('english')
swords.sort()

def hello_ds():
    print("Big hello to you")

def sortSecond(val):
  return val[1]

def euclidean_distance(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for euclidean vectors: {len(vect1)} and {len(vect2)}"

  #rest of your code below
  cur_value = 0
  for i in range(len(vect1)):
    cur_value += np.square(vect1[i]-vect2[i])
  return np.sqrt(cur_value)

def ordered_distances(target_vector:list, crowd_table:dframe, answer_column:str, dfunc:Callable) -> list:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'

  #your code goes here
  crowd1_table = crowd_table.drop(answer_column, 1)
  distance_list = [(index, dfunc(target_vector, row.tolist())) for index, row in crowd1_table.iterrows()]
  distance_list.sort(key = sortSecond)
  return distance_list

def knn(target_vector:list, crowd_table:dframe, answer_column:str, k:int, dfunc:Callable) -> int:
  assert isinstance(target_vector, list), f'target_vector not a list but instead a {type(target_vector)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'

  #your code goes here
  
  top_k = [i for i,d in ordered_distances(target_vector, crowd_table, answer_column, dfunc)[:k]]
  opinions = [crowd_table.loc[i, answer_column] for i in top_k]
  winner = 1 if opinions.count(1) > opinions.count(0) else 0
  return winner

def knn_tester(test_table, crowd_table, answer_column, k, dfunc:Callable) -> dict:
  assert isinstance(test_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(test_table)}'
  assert isinstance(crowd_table, pd.core.frame.DataFrame), f'crowd_table not a dataframe but instead a {type(crowd_table)}'
  assert isinstance(answer_column, str), f'answer_column not a string but instead a {type(answer_column)}'
  assert answer_column in crowd_table, f'{answer_column} is not a legit column in crowd_table - check case and spelling'
  assert crowd_table[answer_column].isin([0,1]).all(), f"answer_column must be binary"
  assert callable(dfunc), f'dfunc not a function but instead a {type(dfunc)}'
  
  #your code here
  collection = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
  for index,row in test_table.iterrows():
    node = (knn(test_table.iloc[index].tolist()[1:], crowd_table, answer_column, k, dfunc), test_table.loc[index, answer_column])
    for i in collection:
      if node == i:
        collection[i] += 1
  return collection

def cm_accuracy(confusion_dictionary: dict) -> float:
  assert isinstance(confusion_dictionary, dict), f'confusion_dictionary not a dictionary but instead a {type(confusion_dictionary)}'
  
  tp = confusion_dictionary[(1,1)]
  fp = confusion_dictionary[(1,0)]
  fn = confusion_dictionary[(0,1)]
  tn = confusion_dictionary[(0,0)]
  
  return (tp+tn)/(tp+fp+fn+tn)

def cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"
  
  #your code here
  top, bot_a, bot_b = 0, 0, 0
  for i in range(len(vect1)):
    top += vect1[i]*vect2[i]
    bot_a += vect1[i]**2
    bot_b += vect2[i]**2
  try:
    return (top/(((bot_a)**(1/2))*((bot_b)**(1/2))))
  except ZeroDivisionError:
    return 0.0

def inverse_cosine_similarity(vect1:list ,vect2:list) -> float:
  assert isinstance(vect1, list), f'vect1 is not a list but a {type(vect1)}'
  assert isinstance(vect2, list), f'vect2 is not a list but a {type(vect2)}'
  assert len(vect1) == len(vect2), f"Mismatching length for vectors: {len(vect1)} and {len(vect2)}"

  normal_result = cosine_similarity(vect1, vect2)
  return 1.0 - normal_result

def bayes(evidence:set, evidence_bag:dict, training_table:dframe) -> tuple:
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"

  #your code here
  label_list = training_table['label'].tolist()

  variation = max(label_list)+1
  
  count = [label_list.count(i) for i in range(variation)]
  p_count = [j/len(training_table) for j in count]

  return tuple(np.multiply(np.prod([np.divide(evidence_bag[key][i], count[i]) for key in evidence]), p_count[i]) for i in range(variation))

def bayes_tester(testing_table:dframe, evidence_bag:dict, training_table:dframe, parser:Callable) -> list:
  assert isinstance(testing_table, pd.core.frame.DataFrame), f'test_table not a dataframe but instead a {type(testing_table)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert callable(parser), f'parser not a function but instead a {type(parser)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert 'text' in testing_table, f'text column is not found in testing_table'

  #your code here
  return [bayes(set(parser(row['text'])), evidence_bag, training_table) for index, row in testing_table.iterrows()]

def get_clean_words(stopwords:list, raw_sentence:str) -> list:
  assert isinstance(stopwords, list), f'stopwords must be a list but saw a {type(stopwords)}'
  assert all([isinstance(word, str) for word in stopwords]), f'expecting stopwords to be a list of strings'
  assert isinstance(raw_sentence, str), f'raw_sentence must be a str but saw a {type(raw_sentence)}'

  sentence = raw_sentence.lower()
  for word in stopwords:
    sentence = re.sub(r"\b"+word+r"\b", '', sentence)  #replace stopword with empty

  cleaned = re.findall("\w+", sentence)  #now find the words
  return cleaned

def robust_bayes(evidence:set, evidence_bag:dict, training_table:dframe, laplace:float=1.0) -> tuple:
  assert isinstance(evidence, set), f'evidence not a set but instead a {type(evidence)}'
  assert isinstance(evidence_bag, dict), f'evidence_bag not a dict but instead a {type(evidence_bag)}'
  assert isinstance(training_table, pd.core.frame.DataFrame), f'training_table not a dataframe but instead a {type(training_table)}'
  assert 'label' in training_table, f'label column is not found in training_table'
  assert training_table.label.dtype == int, f"label must be an int column (possibly wrangled); instead it has type({training_table.label.dtype})"
  assert isinstance(laplace, float), f'laplace not a float but instead a {type(laplace)}'

  #your code here
  label_list = training_table['label'].tolist()

  variation = max(label_list)+1
  v_list = [0 for i in range(variation)]
  
  count = [label_list.count(i) for i in range(variation)]
  p_count = [j/len(training_table) for j in count]

  V = len(evidence_bag)

  return tuple(max(np.multiply(np.prod([np.divide(evidence_bag.get(key, v_list)[i] + laplace, count[i] + V + laplace) for key in evidence]), p_count[i]), 2.2250738585072014e-308) for i in range(variation))

def ordered_animals(target_vector, table):
  #distance_list = [(row.name,euclidean_distance(target_vector, row.tolist())) for i,row in table.iterrows()]  #for list comprehension fans
  distance_list = []
  for i,row in table.iterrows():
    a_vector = row.tolist()
    d = euclidean_distance(target_vector, a_vector)
    distance_list.append((row.name,d))
  return sorted(distance_list, key=lambda pair: pair[1], reverse=False)

def hex_to_int(s:str) -> tuple:
  assert isinstance(s, str)
  assert s[0] == '#'
  assert len(s) == 7

  s = s.lstrip("#")
  return int(s[:2], 16), int(s[2:4], 16), int(s[4:6], 16)

def fast_euclidean_distance(x:narray, y:narray) -> float:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  
  return np.linalg.norm(x-y)

def subtractv(x:narray, y:narray) -> narray:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"

  return np.subtract(x, y)

def addv(x:narray, y:narray) -> narray:
  assert isinstance(x, numpy.ndarray), f"x must be a numpy array but instead is {type(x)}"
  assert len(x.shape) == 1, f"x must be a 1d array but instead is {len(x.shape)}d"
  assert isinstance(y, numpy.ndarray), f"y must be a numpy array but instead is {type(y)}"
  assert len(y.shape) == 1, f"y must be a 1d array but instead is {len(y.shape)}d"
  
  return np.add(x, y)

def meanv_slow(coords):
    # assumes every item in coords has same length as item 0
    sumv = [0] * len(coords[0])
    for item in coords:
        for i in range(len(item)):
            sumv[i] += item[i]
    mean = [0] * len(sumv)
    for i in range(len(sumv)):
        mean[i] = float(sumv[i]) / len(coords)
    return mean

def meanv(matrix: narray) -> narray:
  assert isinstance(matrix, numpy.ndarray), f"matrix must be a numpy array but instead is {type(matrix)}"
  assert len(matrix.shape) == 2, f"matrix must be a 2d array but instead is {len(matrix.shape)}d"

  return np.mean(matrix, axis = 0)

def fast_cosine(v1:narray, v2:narray) -> float:
  assert isinstance(v1, numpy.ndarray), f"v1 must be a numpy array but instead is {type(v1)}"
  assert len(v1.shape) == 1, f"v1 must be a 1d array but instead is {len(v1.shape)}d"
  assert isinstance(v2, numpy.ndarray), f"v2 must be a numpy array but instead is {type(v2)}"
  assert len(v2.shape) == 1, f"v2 must be a 1d array but instead is {len(v2.shape)}d"
  assert len(v1) == len(v2), f'v1 and v2 must have same length but instead have {len(v1)} and {len(v2)}'

  bottom = (norm(v1)*norm(v2))
  return 0.0 if bottom == 0 else np.dot(v1, v2)/bottom

def dict_ordered_distances(space:dict, coord:narray) -> list:
  assert isinstance(space, dict), f"space must be a dictionary but instead a {type(space)}"
  assert isinstance(list(space.values())[0], numpy.ndarray), f"space must have numpy arrays as values but instead has {type(space.values()[0])}"
  assert isinstance(coord, numpy.ndarray), f"coord must be a numpy array but instead is {type(cord)}"
  assert len(list(space.values())[0]) == len(coord), f"space values must be same length as coord"
  assert len(coord) == 3, "coord must be a triple"

  #your code here
  return sorted([(keys, fast_euclidean_distance(values, coord)) for keys, values in space.items()], key = lambda x: x[1])
