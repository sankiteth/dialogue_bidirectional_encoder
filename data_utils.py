# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import gzip
import os
import re
import tarfile
import operator

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
EOS_ID = 1
GO_ID  = 2
UNK_ID = 3

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
  print("In create_vocabulary")
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("processing line %d" % counter)
        text_conversation =line.strip().split("\t")
    
        txt  = text_conversation[0].strip() + " " + text_conversation[1].strip() + " " + text_conversation[2].strip()

        tokens = txt.split()
        for w in tokens:
          word = w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1


      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print("vocab_length={0}".format(len(vocab_list)))

      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")

def initialize_vocabulary(vocabulary_path):
  print("In initialize_vocabulary")
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary):
  words = sentence.strip().split()
  return [vocabulary.get(w, UNK_ID) for w in words]

def batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths

def vocab_experiment():
  vocab = {}
  total = 0
  with gfile.GFile("data/Training_Shuffled_Dataset.txt", mode="r") as f:
    counter = 0
    for line in f:
      counter += 1
      if counter % 100000 == 0:
        print("processing line %d" % counter)
      text_conversation =line.strip().split("\t")
  
      txt  = text_conversation[0].strip() + " " + text_conversation[1].strip() + " " + text_conversation[2].strip()
      tokens = txt.split()
      for w in tokens:
        word = w
        total += 1
        if word in vocab:
          vocab[word] += 1
        else:
          vocab[word] = 1


  print(len(vocab))
  sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)
  print(sorted_vocab[0])

  cum = 0
  percentage = 0
  counter = 0
  for item in sorted_vocab:
    counter += 1
    cum += item[1]

    percentage = cum*100.0/total

    if counter == 3000:
      print("counter={0}, percentage={1}%".format(counter, percentage))

def bucket_experiment():
  buckets = [0,0,0,0,0,0]
  total_examples = 0
  with gfile.GFile("data/Training_Shuffled_Dataset.txt", mode="r") as f:
    for line in f:
      text_conversation =line.strip().split("\t")
  
      txt1  = text_conversation[0].strip()
      txt2  = text_conversation[1].strip()
      txt3  = text_conversation[2].strip()

      for item in [(txt1, txt2), (txt2, txt3)]:
        total_examples += 1
        enc = item[0]
        dec = item[1]
        num_enc = len(enc.split())
        num_dec = len(dec.split())

        if num_enc <= 5 and num_dec <= 10:
          buckets[0] += 1
        elif num_enc <= 10 and num_dec <= 15:
          buckets[1] += 1
        elif num_enc <= 20 and num_dec <= 25:
          buckets[2] += 1
        elif num_enc <= 40 and num_dec <= 50:
          buckets[3] += 1
        elif num_enc <= 100 and num_dec <= 100:
          buckets[4] += 1
        else:
          buckets[5] += 1



  print(buckets)
  print("total examples  ={0}".format(total_examples))
  print("examples covered={0}".format(sum(buckets[:5])))

# if __name__ == '__main__':
#   #create_vocabulary("data/Vocab_file.txt", "data/Training_Shuffled_Dataset.txt", 0)
#   #buckets = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
  

  

