import tensorflow as tf
from model import Seq2SeqModel
import pickle
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('--tf', default = 'cpu', choices = {'gpu','cpu'})
args = parser.parse_args()

tf.reset_default_graph()

word2id = pickle.load( open( "./saved_model/word2id", "rb" ) )
id2word = pickle.load( open( "./saved_model/id2word", "rb" ) )
hidden_size = pickle.load( open( "./saved_model/hidden_size", "rb" ) )

model = Seq2SeqModel(3,word2id,id2word)
model.declare_placeholders()
model.create_embeddings()
model.build_encoder(hidden_size)
model.memory_network(hidden_size)
model.build_decoder(hidden_size,len(word2id.keys()),word2id['^'],word2id['$'])

if(args.tf == 'gpu'):
   session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
else:
   session = tf.Session()

session.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(session,"./saved_model/model.ckpt")
print("")
print("Welcome to chat with chatbot. The default memory of bot is 3 dialog entries. Type \c to reset memory, type \c and number (e.g. \c 4) to clear memory and declare new size of memory. Type \q to exit program")

inp = input("You: ")
while inp != "\q":
  inp = inp.strip() 
  inp_list = inp.split()
  if inp_list[0] == "\c":
    try:   
      model.empty_memory(int(inp_list[1]))
      print("Memory is reset. New memory:",int(inp_list[1]))
      inp = input("You: ")  
    except:
      model.empty_memory(3)
      print("Memory is reset. New memory: 3")
      inp = input("You: ")
  else:
    c = model.predict_sentence(inp,session)
    c = re.sub('\$','', c)
    print("Bot:",c)
    inp = input("You: ")
