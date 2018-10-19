import tensorflow as tf
from utils import *

class Seq2SeqModel(object):
   
  def __init__(self,size_memory,word2id,id2word):
    self.memory_list = []
    self.size_memory = size_memory
    self.begin = 0
    self.word2id = word2id
    self.id2word = id2word

  def empty_memory(self,size_memory):
    self.memory_list = []
    self.size_memory = size_memory
    self.begin = 0
    
  def add_to_memory(self,sequence):
    if self.size_memory > 0:
     if len(self.memory_list) < self.size_memory:
       self.memory_list.append(sequence)
     else:
        self.memory_list.pop(0)
        self.memory_list.append(sequence)  
    
  def return_memory(self):
      return self.memory_list 
  
  def declare_placeholders(self):

    self.input_batch = tf.placeholder(shape=(None, None), dtype=tf.int64, name='input_batch')
    self.input_batch_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='input_batch_lengths')


    self.ground_truth = tf.placeholder(shape=(None,None), dtype=tf.int64, name='ground_truth')
    self.ground_truth_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='ground_truth_lengths')
   
    self.memory_batch = tf.placeholder(shape=(None,None), dtype=tf.int64, name='memory')
    self.memory_batch_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='memory_lengths')

    self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])

    self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[])
    self.mem_size = tf.placeholder(dtype=tf.int64, shape=[])
    self.mode = tf.placeholder(dtype=tf.int64, shape=[])


  def create_embeddings(self):

    random_initializer = tf.random_uniform((len(self.word2id.keys()), 300), -1.0, 1.0)
    self.embeddings = tf.Variable(initial_value = random_initializer, dtype=tf.float32)

    self.input_batch_embedded = tf.nn.embedding_lookup(self.embeddings,self.input_batch)

  def build_encoder(self, hidden_size):

    encoder_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size),input_keep_prob = self.dropout_ph,dtype = tf.float32)

    self.encoder_out, self.final_encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.input_batch_embedded, sequence_length = self.input_batch_lengths,dtype=tf.float32) 

  def build_decoder(self, hidden_size, vocab_size, start_symbol_id, end_symbol_id):


    start_tokens = tf.fill([tf.shape(self.input_batch)[0]], start_symbol_id)
    ground_truth_as_input = tf.concat([tf.cast(tf.expand_dims(start_tokens, 1),tf.int64), self.ground_truth], 1)
   
    self.ground_truth_embedded = tf.nn.embedding_lookup(self.embeddings,ground_truth_as_input)

    train_helper = tf.contrib.seq2seq.TrainingHelper(self.ground_truth_embedded,
                                                     self.ground_truth_lengths)
    infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings,start_tokens,end_symbol_id)  
 
    def decode(helper, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = hidden_size, memory = self.encoder_out, memory_sequence_length = self.input_batch_lengths)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size,reuse=reuse),input_keep_prob = self.dropout_ph,dtype = tf.float32)
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                 cell = decoder_cell,
                 attention_mechanism = attention_mechanism,
                 attention_layer_size = hidden_size)


            decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(decoder_cell, vocab_size, reuse=reuse)

            state = tf.cond(self.mem_size > 0, lambda: decoder_cell.zero_state(tf.shape(self.input_batch)[0], tf.float32).clone(cell_state= tf.add(self.final_encoder_state,self.memory_out)), lambda: decoder_cell.zero_state(tf.shape(self.input_batch)[0], tf.float32).clone(cell_state=self.final_encoder_state))

            decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,helper=helper,initial_state=state)

            def dynamic_decode_training():
             outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True)    
             return outputs
            def dynamic_decode_infer():
             outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True,maximum_iterations = 7)    
             return outputs
            return tf.cond(self.mode < 3,dynamic_decode_training,dynamic_decode_infer)
       
    self.train_outputs = decode(train_helper, 'decode')
    self.train_predictions = self.train_outputs.sample_id
    self.infer_outputs = decode(infer_helper, 'decode', reuse=True)
    self.infer_predictions = self.infer_outputs.sample_id
        
  def memory_network(self,hidden_size):
     encoder_cell_input = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size),input_keep_prob = self.dropout_ph,dtype = tf.float32)  
     encoder_cell_memory = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(hidden_size),input_keep_prob = self.dropout_ph,dtype = tf.float32)

     self.memory_batch_embedded = tf.nn.embedding_lookup(self.embeddings,self.memory_batch)

     self.dense_layer = tf.layers.Dense(hidden_size)
     _, self.final_memory_encoder_state = tf.nn.dynamic_rnn(encoder_cell_memory, self.memory_batch_embedded, sequence_length = self.memory_batch_lengths,dtype=tf.float32,scope ="rnn2")
     _, self.final_input_encoder_state = tf.nn.dynamic_rnn(encoder_cell_input, self.input_batch_embedded, sequence_length = self.input_batch_lengths, dtype=tf.float32,scope ="rnn3")

     
     def dense_layer(summation):
        result = self.dense_layer(tf.reshape(summation,[-1,hidden_size]))
        return result

     def train():  
      def train_with_memory():  
       final_memory_encoder_state = tf.reshape(self.final_memory_encoder_state,[tf.shape(self.input_batch)[0],-1,hidden_size])
       matrix_product = tf.matmul(final_memory_encoder_state[:][:][0],  tf.expand_dims(self.final_input_encoder_state[0],1) )
       P = tf.nn.softmax(tf.transpose(matrix_product))
       H = tf.matmul(P, final_memory_encoder_state[:][:][0])
       def condition(i,H):
         return tf.less(i,tf.shape(self.input_batch)[0])
       def body(num,H):
         matrix_product = tf.matmul(final_memory_encoder_state[:][:][num],  tf.expand_dims(self.final_input_encoder_state[num],1))
         P = tf.nn.softmax(tf.transpose(matrix_product))
         H = tf.concat([H,tf.matmul(P, final_memory_encoder_state[:][:][num])],0)

         num += 1  
         return num,H  
       _, H1 = tf.while_loop(condition, body, loop_vars = [1,H],shape_invariants = [tf.constant(1).get_shape(),tf.TensorShape([None, hidden_size])])
       summation = tf.add(self.final_input_encoder_state,H1) 

       memory_out = dense_layer(summation) 
       return memory_out 
      def train_without_memory():
       return self.final_input_encoder_state
    
      L = tf.cond(self.mode < 2,train_with_memory,train_without_memory)
      return L  
     def predict(): 
       def predict_with_memory(): 
        matrix_product = tf.matmul(self.final_input_encoder_state, tf.transpose(self.final_memory_encoder_state))
        P = tf.nn.softmax(matrix_product)
        H = tf.matmul(P,self.final_memory_encoder_state)
        summation = tf.add(self.final_input_encoder_state,H) 
        memory_out = dense_layer(summation)
        return memory_out
       def predict_without_memory():
        return self.final_input_encoder_state
       a = tf.cond(self.mode > 4,predict_with_memory,predict_without_memory)
       return a
       
     self.memory_out = tf.cond(self.mode < 3,train,predict)
  
  def compute_loss(self):
     weights = tf.cast(tf.sequence_mask(self.ground_truth_lengths), dtype=tf.float32)
     self.sliced_ground_truth = tf.slice(self.ground_truth, [0, 0], [-1, tf.shape(self.train_outputs.rnn_output)[1]])   
     self.loss = tf.contrib.seq2seq.sequence_loss(self.train_outputs.rnn_output,self.sliced_ground_truth,weights)

   
  def perform_optimization(self):
     def clip_gradients_by_l2(grads_and_vars):
        modified_grads_vars = [(tf.clip_by_norm(gv[0],20), gv[1]) for gv in grads_and_vars]
        return modified_grads_vars
     self.train_op = tf.contrib.layers.optimize_loss(self.loss,optimizer="Adam",learning_rate = self.learning_rate_ph,global_step=tf.train.get_global_step(),clip_gradients=clip_gradients_by_l2)

  def train_on_batch(self, session, X, X_seq_len, Y, Y_seq_len, mem_batch, mem_batch_len,memory, learning_rate, dropout_keep_probability,mode):
     feed_dict = {}
     if mode == 1:
      feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len,
            self.memory_batch: mem_batch,
            self.memory_batch_lengths: mem_batch_len,
            self.mem_size: memory,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_probability,
            self.mode: 1
       
      }
     else:
      feed_dict = {
            self.input_batch: X,
            self.input_batch_lengths: X_seq_len,
            self.ground_truth: Y,
            self.ground_truth_lengths: Y_seq_len,
            self.memory_batch: mem_batch,
            self.memory_batch_lengths: mem_batch_len,          
            self.mem_size: memory,
            self.learning_rate_ph: learning_rate,
            self.dropout_ph: dropout_keep_probability,
            self.mode: 2       
      }
    

     _, loss, _ = session.run([
            self.train_predictions,
            self.loss,
            self.train_op], feed_dict=feed_dict)
     return loss

  def predict_sentence(self,input_sequence,session):
     encoded_sequence, encoded_sequence_len = sentence_to_ids(input_sequence,self.word2id)
     if self.begin == 0:
        feed_dict = {
        self.input_batch: [encoded_sequence],
        self.input_batch_lengths : [encoded_sequence_len],
        self.mem_size : 0,
        self.mode : 4,
        self.memory_batch : [[0]],
        self.memory_batch_lengths : [1]
    
        }

        prediction = session.run([self.infer_predictions], feed_dict=feed_dict)

        self.add_to_memory(input_sequence)
        self.add_to_memory(' '.join(ids_to_sentence(prediction[0][0], self.id2word)))
        self.begin = 1         
     else:
        encoded_memory, encoded_memory_len = batch_to_ids(self.memory_list,self.word2id)
        feed_dict = {
        self.input_batch: [encoded_sequence],
        self.input_batch_lengths : [encoded_sequence_len],
        self.mem_size : len(self.memory_list),
        self.memory_batch: encoded_memory,
        self.memory_batch_lengths: encoded_memory_len,
        self.mode : 5   
        }
        prediction = session.run([self.infer_predictions], feed_dict=feed_dict)
        self.add_to_memory(input_sequence)
        self.add_to_memory(' '.join(ids_to_sentence(prediction[0][0], self.id2word)))
     return ' '.join(ids_to_sentence(prediction[0][0], self.id2word))   
