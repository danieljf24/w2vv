import numpy as np
from basic.common import printStatus
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences


INFO = __file__

class W2VV_MS_pred(object):
    def __init__(self, model_path, weight_path):
        self.model = model_from_json(open(model_path).read())
        self.model.load_weights(weight_path)
        # any loss ang optimizers are ok
        self.model.compile(loss='mse', optimizer='sgd')
        printStatus(INFO + '.' + self.__class__.__name__, 'loaded a trained Word2VisualVec model successfully')

    def predict_one(self, text_vec, text_vec_2):
        text_embed_vec = self.model.predict([np.array([text_vec]), np.array([text_vec_2])])
        return text_embed_vec[0]

    def predict_batch(self, text_vec_batch, text_vec_batch_2):
        text_embed_vecs = self.model.predict([np.array(text_vec_batch), np.array(text_vec_batch_2)])
        return text_embed_vecs




def pred_mutual_error_ms(img_list, sents, predictor, text2vec, bow2vec, w2v2vec, img_feats, losser, batch_size=10000, opt=None):

    print "embedding all sentences ..."
    text_emd = []
    flag_list = []
    for start in range(0, len(sents), batch_size):
        end = min(len(sents), start+batch_size)
        raw_sent_batch = sents[start: end]
        text_batch = []
        text_rnn_batch = []
        for sent in raw_sent_batch:
            sent_vec = text2vec.mapping(sent)
            sent_bow_vec = bow2vec.mapping(sent)
            sent_w2v_vec = w2v2vec.mapping(sent)
            if sent_vec is not None and sent_bow_vec is not None and sent_w2v_vec is not None:
                text_rnn_batch.append(sent_vec)
                text_batch.append(list(sent_bow_vec) + list(sent_w2v_vec))
                flag_list.append([1])
            else:
                print sent
                print '[error]'
                text_rnn_batch.append([0])
                text_batch.append([0]*(bow2vec.ndims+w2v2vec.ndims))
                flag_list.append([0])
        text_rnn_batch = pad_sequences(text_rnn_batch, maxlen=opt.sent_maxlen,  truncating='post')

        text_emd_batch = predictor.predict_batch(text_rnn_batch, text_batch)
        text_emd.extend(text_emd_batch)
    assert len(text_emd) == len(sents)


    print "embedding all images ..."
    img_emd = []
    for img in img_list:
        img_emd.append(img_feats.read_one(img))
    assert len(img_emd) == len(img_list)

    assert len(text_emd) == opt.n_caption*len(img_list), '%d != %d' % (len(text_emd), opt.n_caption*len(img_list))


    print("matching image and text ...")
    text_batch_size = 1000
    counter = 0
    all_errors = []
    for start in range(0, len(sents), text_batch_size):
        end = min(len(sents), start+text_batch_size)
        text_emd_batch = text_emd[start: end]

        errorlist_batch = losser.calculate(text_emd_batch, img_emd)
        all_errors.append(errorlist_batch)

    all_errors = np.concatenate(all_errors, axis=0)
    print all_errors.shape

    flag_matrix = np.tile(flag_list, len(img_list))
    all_errors = all_errors*flag_matrix + (1-flag_matrix)*1000

    return all_errors
