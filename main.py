import pprint
import tensorflow as tf
from data import *
from model_with_mask import MemN2N

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 300, "internal state dimension [300]")
flags.DEFINE_integer("lindim", 100, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 3, "number of hops [7]")
flags.DEFINE_integer("batch_size", 1, "batch size to use during training [1]")
flags.DEFINE_integer("nepoch", 20, "number of epoch to use during training [100]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.01, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [100]")
flags.DEFINE_string("pretrain_embeddings", "glove-common_crawl_840",
                    "pre-trained word embeddings [glove-wikipedia_gigaword, glove-common_crawl_48, glove-common_crawl_840]")
flags.DEFINE_string("train_data", "data/Restaurants_Train_v2.xml.seg", "train gold data set path [./data/Laptops_Train.xml.seg]")
flags.DEFINE_string("test_data", "data/Restaurants_Test_Gold.xml.seg", "test gold data set path [./data/Laptops_Test_Gold.xml.seg]")
flags.DEFINE_boolean("show", False, "print progress [False]")

FLAGS = flags.FLAGS


def get_idx2word(word2idx):
    idx2word = {}
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return idx2word


def main(_):
  source_word2idx, target_word2idx, word_set = {}, {}, {}
  max_sent_len = -1
  
  max_sent_len = get_dataset_resources(FLAGS.train_data, source_word2idx, target_word2idx, word_set, max_sent_len)
  max_sent_len = get_dataset_resources(FLAGS.test_data, source_word2idx, target_word2idx, word_set, max_sent_len)
  #embeddings = load_embedding_file(FLAGS.pretrain_embeddings, word_set)

  embeddings = init_word_embeddings(FLAGS.pretrain_embeddings, word_set, FLAGS.edim)
  train_data = get_dataset(FLAGS.train_data, source_word2idx, target_word2idx, embeddings)
  test_data = get_dataset(FLAGS.test_data, source_word2idx, target_word2idx, embeddings)

  print("train data size - ", len(train_data[0]))
  print("test data size - ", len(test_data[0]))

  print("max sentence length - ", max_sent_len)
  FLAGS.pad_idx = source_word2idx['<pad>']
  FLAGS.nwords = len(source_word2idx)
  FLAGS.mem_size = max_sent_len

  pp.pprint(flags.FLAGS.__flags)

  print('loading pre-trained word vectors...')
  print('loading pre-trained word vectors for train and test data')
  
  FLAGS.pre_trained_context_wt, FLAGS.pre_trained_target_wt = get_embedding_matrix(embeddings, source_word2idx,  target_word2idx, FLAGS.edim)

  source_idx2word, target_idx2word = get_idx2word(source_word2idx), get_idx2word(target_word2idx)

  with tf.Session() as sess:
    model = MemN2N(FLAGS, sess, source_idx2word, target_idx2word)
    model.build_model()
    model.run(train_data, test_data)  

if __name__ == '__main__':
  tf.app.run()  