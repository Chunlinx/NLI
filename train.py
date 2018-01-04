from model import model
import numpy as np
import tensorflow as tf
from preprocess import load_train_data,load_test_data
from glove import word_embedings
import random,pickle
batch_size = 512
def save_variable(a,path):
    f = open(path,'wb')
    pickle.dump(a,f)
    f.close()

def batch_iter(data, batch_size, epochs, Isshuffle=True):
    ## check inputs
    assert isinstance(batch_size,int)
    assert isinstance(epochs,int)
    assert isinstance(Isshuffle,bool)

    num_batches = int((len(data)-1)/batch_size)
    ## data padded
    # data = np.array(data+data[:2*batch_size])
    data_size = len(data)
    print("size of data"+str(data_size)+"---"+str(len(data)))
    for ep in range(epochs):
        if Isshuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            yield shuffled_data[start_index:end_index]

def train(m,data_1,data_2,data_len_1,data_len_2,train_label,epochs=800,learning_rate=0.001,check_point=500):

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads_and_vars = optimizer.compute_gradients(m.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    sess = tf.Session(config=session_conf)
    saver = tf.train.Saver()
    ## intialize
    sess.run(tf.global_variables_initializer())
    train_data = list(zip(data_1,data_2,data_len_1,data_len_2,train_label))
    batches = batch_iter(train_data,batch_size=batch_size,epochs=epochs,Isshuffle=False)
    ## run the graph
    print("\n")
    i = 0
    max_acc = -1
    for batch in batches:
        x_1,x_2,len_1,len_2,y = zip(*batch)
        x_1 = np.array(x_1)
        x_2 = np.array(x_2)
        len_1 = np.array(len_1)
        len_2 = np.array(len_2)
        y = np.array(y)
        feed_dict = {
            m.sent1           : x_1,
            m.sent2           : x_2,
            m.sent1_length    : len_1,
            m.sent2_length    : len_2,
            m.labels          : y,
            m.dropout_keep_prob         : 0.5
        }
        _,loss,accuracy = sess.run([train_op,m.loss,m.acc],feed_dict=feed_dict)
        print("step - "+str(i)+"    loss is " + str(loss)+" and accuracy is "+str(accuracy))
        sum_acc = 0
        sum_loss = 0

        if i%check_point == 0 and i > 0:
            j = 0
            test_batches = batch_iter(list(zip(test_data_1,test_data_2,test_data_len_1,test_data_len_2,test_label)), batch_size=batch_size, epochs=1,Isshuffle = False)
            for test_batch in test_batches:
                x_1,x_2,len_1,len_2,y = zip(*test_batch)
                x_1 = np.array(x_1)
                x_2 = np.array(x_2)
                len_1 = np.array(len_1)
                len_2 = np.array(len_2)

                y = np.array(y)
                feed_dict = {
                    m.sent1        : x_1,
                    m.sent2        : x_2,
                    m.sent1_length : len_1,
                    m.sent2_length : len_2,
                    m.labels       : y,
                    m.dropout_keep_prob      : 1.0
                }

                loss, accuracy = sess.run([m.loss, m.acc], feed_dict=feed_dict)
                sum_acc += accuracy
                sum_loss += loss
                j += 1
            print(" test loss is " + str(sum_loss / j) + " and test-accuracy is " + str(sum_acc / j))
            if sum_acc/j > max_acc:
                max_acc = sum_acc/j
                save_path = "saved_models/model-" + str(i)
                saver.save(sess, save_path=save_path)
                print("Model saved to " + save_path)

        i += 1
    return sess

# this will load data from default path
word_vecs = word_embedings(debug=False)

embedding_size = 300

# train_paths = []
# test_paths = []
train_path = r'C:\Users\pravi\PycharmProjects\NLI\data\snli_1.0_train.txt'
dev_path = r'C:\Users\pravi\PycharmProjects\NLI\data\snli_1.0_dev.txt '
# test_path = r'C:\Users\pravi\PycharmProjects\NLI\data\snli_1.0_test.txt'
res = load_train_data(train_path)
save_variable(res,path=r'C:\Users\pravi\PycharmProjects\NLI\data_pickles\data')
print("done")

train_data_1 = res['data_1']
train_data_2 = res['data_2']
train_label = res['labels']
train_data_len_1 = res['data_length_1']
train_data_len_2 = res['data_length_2']
word2Id = res['word2Id']

words_data_list = word2Id.keys()
Id2Word  = res['Id2Word']
max_sequence_length = res['max_sequence_length']
total_classes = res['total_classes']


test_res = load_test_data(dev_path,word2Id,Id2Word,max_sequence_length)
test_data_1 = test_res['data_1']
test_data_2 = test_res['data_2']
test_label = test_res['labels']
test_data_len_1 = test_res['data_length_1']
test_data_len_2 = test_res['data_length_2']
word2Id = test_res['word2Id']


Id2Vec = np.zeros([len(Id2Word.keys()),embedding_size])
words_list = word_vecs.word2vec.keys()
for i in range(len(Id2Word.keys())):
    word = Id2Word[i]
    if word in words_list:
        Id2Vec[i,:] = word_vecs.word2vec[word]
    else:
        Id2Vec[i, :] = word_vecs.word2vec['unknown']

m = model(
          max_sequence_length=max_sequence_length,
          total_classes=total_classes,
          embedding_size=300,
          id2Vecs= Id2Vec,
          batch_size=batch_size
        )

train(m,train_data_1,train_data_2,train_data_len_1,train_data_len_2,train_label,learning_rate=0.002)