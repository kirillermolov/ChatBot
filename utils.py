import random
import re

def remove_characters(line):
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    line = re.sub('\^','', line)
    line = re.sub('\#','', line)
    line = re.sub('\$','', line)
    line = re.sub('\*','', line)
    return line

def basic_tokenizer(line):
    words = []
    _WORD_SPLIT = re.compile("([.,!?\-<>:;)(])")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            words.append(token)
    return words


def create_vocabulary(max_vocabulary_size,filename):
 word2id = {}
 lines_movie = open(filename,encoding='iso-8859-1').read().splitlines()
 num_iter = 0
 word_index = 0
 while len(word2id.keys()) < max_vocabulary_size or num_iter < max_vocabulary_size*2:
    line = random.choice(lines_movie)
    movie_line = line.split(' +++$+++ ')
    if len(movie_line) == 5:
          edited_movie_line = movie_line[4].strip().lower() 
          edited_movie_line = remove_characters(edited_movie_line)  
          tokenized_movie_line = basic_tokenizer(edited_movie_line)
          for item in tokenized_movie_line:
            if item not in word2id.keys():
                   word2id[item] = word_index
                   word_index+=1
          num_iter += 1
 start_symbol = '^'
 word2id[start_symbol] = word_index
 word_index += 1
 end_symbol = '$'
 word2id[end_symbol] = word_index
 word_index += 1
 padding_symbol = '*'
 word2id[padding_symbol] = word_index
 word_index += 1
 unknown_symbol = '#'
 word2id[unknown_symbol] = word_index
               
 return word2id

def ids_to_sentence(ids, id2word):
    return [id2word[i] for i in ids]

def get_movie_lines(filename):
 f = open(filename,'r', encoding='iso-8859-1')
 get_movie_lines = {}
 for i,line in enumerate(f):
        movie_line = line.split(' +++$+++ ')
        if len(movie_line) == 5:
          edited_movie_line = movie_line[4].strip().lower() 
          edited_movie_line = remove_characters(edited_movie_line)  
          get_movie_lines[movie_line[0]] = edited_movie_line
 f.close() 
 return get_movie_lines


def prepare_dialog(filename):
 f = open(filename,'r', encoding='iso-8859-1')
 memory_n = {}
 input_sequence = {}
 output_sequence = {}
 for line in f:
   k = line.split('+++$+++ ')
   k[3] = re.sub('\[', '', k[3])
   k[3] = re.sub('\]', '', k[3])
   k[3] = re.sub('\'', '', k[3])
   k[3] = k[3].strip()
   l = k[3].split(", ")
   length = len(l)
   if length == 2:
      if 0 in input_sequence.keys():  
        input_sequence[0].append(l[0])
        output_sequence[0].append(l[1])
      else:
        input_sequence[0] = [l[0]]
        output_sequence[0] = [l[1]]
   elif length > 2:
      if 0 in input_sequence.keys():  
        input_sequence[0].append(l[0])
        output_sequence[0].append(l[1])
      else:
        input_sequence[0] = [l[0]]
        output_sequence[0] = [l[1]]
      position = 0
      while position <= length-2:
        memory = 1
        while position+memory-1 < length-2:
         i = 0
         memory_list = []
         while i < memory:
          memory_list.append(l[position+i])
          i+=1
         if memory in memory_n.keys():
          memory_n[memory].append(memory_list)
          input_sequence[memory].append(l[position+i])
          output_sequence[memory].append(l[position+i+1])
         else:
          memory_n[memory] = [memory_list]
          input_sequence[memory] = [l[position+i]]
          output_sequence[memory] = [l[position+i+1]]
         memory += 1
        position += 1
 f.close()
 return input_sequence,output_sequence,memory_n
 

def sentence_to_ids(sentence, word2id, padded_len = 0):
    sent_ids = []
    sentence = remove_characters(sentence)
    sentence = basic_tokenizer(sentence)
    if padded_len == 0:
        il = len(sentence)
    else:
        il = min(len(sentence),padded_len-1)
    for x in range(0,il):
      try:  
        sent_ids.append(word2id[sentence[x]])
      except:
        sent_ids.append(word2id['#'])

    sent_ids.append(word2id['$'])
    sent_len = len(sent_ids)
    while len(sent_ids) < padded_len:
        sent_ids.append(word2id['*'])  
    return sent_ids, sent_len
   
def batch_to_ids(sentences, word2id, max_len=0):

    if max_len > 0:
     max_len_in_batch = min(max(len(basic_tokenizer(s)) for s in sentences) + 1, max_len)
    else:
     max_len_in_batch = max(len(basic_tokenizer(s)) for s in sentences) + 1   
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len


def generate_batches_no_memory(input_samples,output_samples,get_movie_lines,batch_size=64):
    X,Y = [],[]
    for i, (x,y) in enumerate(zip(input_samples,output_samples), 1):
        X.append(get_movie_lines[x])
        Y.append(get_movie_lines[y])
        if i % batch_size == 0:
            yield X,Y
            X,Y= [],[]
    if X and Y:
        yield X,Y
   
def generate_batches_memory(input_samples,output_samples,memory_samples,get_movie_lines,batch_size=64):
    input_seq, output_seq, memory = [], [], []
    for i, (x,y,z) in enumerate(zip(input_samples,output_samples,memory_samples), 1):
     input_seq.append(get_movie_lines[x])
     output_seq.append(get_movie_lines[y])
     for pos in range(0,len(z)):  
      memory.append(get_movie_lines[z[pos]])
     if i % batch_size == 0:
            yield input_seq, output_seq, memory
            input_seq, output_seq, memory = [], [], []
    if input_seq and output_seq and memory:
        yield input_seq, output_seq, memory


