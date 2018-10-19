from model import Seq2SeqModel
import tensorflow as tf
import numexpr
import argparse
import pickle
from utils import *
parser = argparse.ArgumentParser()
parser.add_argument('--tf', default = 'cpu', choices = {'gpu','cpu'})
parser.add_argument('--maxvocab',default ='50000')
parser.add_argument('--hidden',default ='256')
parser.add_argument('--epoch',default ='10')
parser.add_argument('--batch',default ='128')
parser.add_argument('--maxlen',default ='16')
parser.add_argument('--rate',default ='0.001')
parser.add_argument('--dropout', default = '0.5')
parser.add_argument('--decrate',default = '0.5')
parser.add_argument('--decstep',default = '5')

args = parser.parse_args()

word2id = create_vocabulary(int(args.maxvocab),'./data/cornell movie-dialogs corpus/movie_lines.txt')
id2word = {i:symbol for symbol, i in word2id.items()}

movie_lines = get_movie_lines('./data/cornell movie-dialogs corpus/movie_lines.txt')
input_sequence,output_sequence,memory_n = prepare_dialog('./data/cornell movie-dialogs corpus/movie_conversations.txt')

tf.reset_default_graph()
model = Seq2SeqModel(3,word2id,id2word)
model.declare_placeholders()
model.create_embeddings()
model.build_encoder(int(args.hidden))
model.memory_network(int(args.hidden))
model.build_decoder(int(args.hidden),len(word2id.keys()),word2id['^'],word2id['$'])
model.compute_loss()
model.perform_optimization()

if(args.tf == 'gpu'):
   session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
else:
   session = tf.Session()
session.run(tf.global_variables_initializer())
 
n_epochs = int(args.epoch) 
batch_size = int(args.batch)
max_len = int(args.maxlen)
learning_rate = float(args.rate)
dropout_keep_probability = float(args.dropout)

print('Start training... \n')
number_epoch = 1
for epoch in range(n_epochs):
    if(number_epoch%int(args.decstep)==0):
        learning_rate = float(args.decrate)*learning_rate    
    print('Train: epoch', epoch + 1)
    n_step = int(len(input_sequence[0])/batch_size)
    for n_iter, (X_batch, Y_batch) in enumerate(generate_batches_no_memory(input_sequence[0],output_sequence[0],movie_lines,batch_size=batch_size)):
     input_seq, x_len1 = batch_to_ids(X_batch, word2id, max_len=max_len)
     output_seq, y_len1 = batch_to_ids(Y_batch, word2id, max_len=max_len)
     loss = model.train_on_batch(session, input_seq,x_len1,output_seq,y_len1,[[0,0]],[1],0,learning_rate, dropout_keep_probability,2)
     if n_iter % 200 == 0:
      print("Movie dialogue training, Memory size: 0, Epoch: [%d/%d], step: [%d/%d], loss: %f" % (epoch + 1, n_epochs, n_iter + 1,n_step, loss))

    for memory in memory_n.keys():
     n_step = int(len(input_sequence[memory])/batch_size)
     for n_iter, (X_batch, Y_batch, Z_batch) in enumerate(generate_batches_memory(input_sequence[memory],output_sequence[memory], memory_n[memory],movie_lines, batch_size=batch_size)):
        input_seq, x_len1 = batch_to_ids(X_batch, word2id, max_len = max_len)
        output_seq, y_len1 = batch_to_ids(Y_batch, word2id, max_len = max_len)
        memory_seq, mem_len = batch_to_ids(Z_batch,word2id, max_len = max_len)

        loss = model.train_on_batch(session, input_seq,x_len1,output_seq,y_len1,memory_seq,mem_len,memory,learning_rate, dropout_keep_probability,1)
        if n_iter % 200 == 0:
            print("Movie dialogue training, Memory size: %d, Epoch: [%d/%d], step: [%d/%d], loss: %f" % (memory ,epoch + 1, n_epochs, n_iter + 1,n_step, loss))

    number_epoch+=1             
print('\n...training finished.')
saver = tf.train.Saver()
saver.save(session,'./saved_model/model.ckpt')
pickle.dump( word2id, open( "./saved_model/word2id", "wb" ) )
pickle.dump( id2word, open( "./saved_model/id2word", "wb" ) )
pickle.dump( int(args.hidden), open( "./saved_model/hidden_size", "wb" ) )
