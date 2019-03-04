__author__ = 'Bingqing Wei'
from model3_batch.model import *

class ModelV3(ProteinModel):
    def build_ta_pred(self, msa_tensor, amino_tensor, secondary_tensor,
                      n_words_aa, n_words_q8):
        amino_embed = Embedding(input_dim=n_words_aa, output_dim=128)(amino_tensor)
        secondary_embed = Embedding(input_dim=n_words_q8, output_dim=64)(secondary_tensor)

        x = Concatenate(axis=-1)([msa_tensor, secondary_embed, amino_embed])
        return unet(x)


class ModelV2(ProteinModel):
    def build_ta_pred(self, msa_tensor, amino_tensor, secondary_tensor,
                      n_words_aa, n_words_q8):

        amino_embed = Embedding(input_dim=n_words_aa, output_dim=128)(amino_tensor)
        secondary_embed = Embedding(input_dim=n_words_q8, output_dim=64)(secondary_tensor)

        x = Concatenate(axis=-1)([msa_tensor, secondary_embed, amino_embed])
        x = Bidirectional(GRU(units=128, return_sequences=True,
                              recurrent_dropout=0.1, recurrent_activation='relu'))(x)
        ta_pred = TimeDistributed(Dense(units=3, activation=activations.tanh))(x)
        return ta_pred


class ModelV1(ProteinModel):
    def build_ta_pred(self, msa_tensor, amino_tensor, secondary_tensor,
                      n_words_aa, n_words_q8):
        def build_attention_block(in_l, in_r):
            #x = tf.tensordot(in_l, in_r, axes=-1)
            x = in_l * in_r
            x = tf.nn.softmax(x)
            #x = tf.tensordot(in_l, x)
            x = in_l * x
            return tf.concat([in_r, x], axis=-1)

        amino_embed = Embedding(input_dim=n_words_aa, output_dim=128)(amino_tensor)
        secondary_embed = Embedding(input_dim=n_words_q8, output_dim=64)(secondary_tensor)

        x = Concatenate(axis=-1)([msa_tensor, secondary_embed, amino_embed])

        l1 = Bidirectional(LSTM(64, activation=activations.relu, return_sequences=True))(x)
        l2 = LSTM(128, activation=activations.relu, return_sequences=True)(l1)
        l3 = LSTM(128, activation=activations.relu, return_sequences=True)(l2)
        l4 = LSTM(128, activation=activations.relu, return_sequences=True)(l3)
        l5 = LSTM(128, activation=activations.relu, return_sequences=True)(l4)

        lstms = [l1, l2, l3, l4, l5]
        attention_blocks = []
        for i in range(len(lstms)):
            for j in range(i + 1, len(lstms)):
                attention_blocks.append(build_attention_block(lstms[i], lstms[j]))
        x = Add()(attention_blocks)
        #ta_pred = LSTM(3, activation=activations.tanh, return_sequences=True)
        ta_pred = TimeDistributed(Dense(units=3, activation=activations.tanh))(x)
        return ta_pred


def train(model_class):
    np.set_printoptions(threshold=np.nan)
    loader = load_loader(mode='train', max_size=3000, work_dir=os.path.join('..', 'data'))
    model = model_class(work_dir='./data', per_process_gpu_memory_fraction=0.7,
                        n_words_aa=loader.n_words_aa, n_words_q8=loader.n_words_q8)
    model.train(X=loader.X, Y=loader.Y, test_size=0.2, batch_size=32,
                nb_epochs=100, verbose=True)

def predict(model_class, ckpt_fname):
    loader = load_loader(mode='test', max_size=None, work_dir=os.path.join('..', 'data'))
    model = model_class(work_dir=os.path.join('.', 'data'),
                         per_process_gpu_memory_fraction=0.7,
                         n_words_aa=loader.n_words_aa, n_words_q8=loader.n_words_q8)
    model.predict(loader.X, os.path.join('.', 'data', ckpt_fname),
                  save_dir=os.path.join('..', 'data'))

if __name__ == '__main__':
    #predict(ckpt_fname='model.ckpt-8')
    train(ModelV2)
