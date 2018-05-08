import os
import numpy as np

from simpleknn.bigfile import BigFile
from basic.common import printStatus
from basic.constant import ROOT_PATH as rootpath

from text import clean_str, clean_str_filter_stop

INFO = __file__


class Text2Vec:

    def __init__(self, datafile, ndims = 0, L1_normalize = 0, L2_normalize = 0):
        printStatus(INFO + '.' + self.__class__.__name__, 'initializing ...')
        self.datafile = datafile
        self.ndims = ndims
        self.L1_normalize = L1_normalize
        self.L2_normalize = L2_normalize


        assert type(L1_normalize) == int
        assert type(L2_normalize) == int
        assert (L1_normalize + L2_normalize) <= 1

    def embedding(self, query):
        vec = self.mapping(query)
        if vec is not None:
            vec = np.array(vec)
        return vec

    def do_L1_norm(self, vec):
        L1_norm = np.linalg.norm(vec, 1)
        return 1.0 * np.array(vec) / L1_norm

    def do_L2_norm(self, vec):
        L2_norm = np.linalg.norm(vec, 2)
        return 1.0 * np.array(vec) / L2_norm





# word2vec + average pooling
class AveWord2Vec(Text2Vec):

    # datafile: the path of pre-trained word2vec data
    def __init__(self, datafile, ndims = 0, L1_normalize = 0, L2_normalize = 0):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)
        self.word2vec = BigFile(datafile)
        if ndims != 0:
            assert self.word2vec.ndims == self.ndims, "feat dimension is not match %d != %d" % (self.word2vec.ndims, self.ndims)
        else:
            self.ndims = self.word2vec.ndims


    def preprocess(self, query, clear):
        if clear:
            words = clean_str(query)
        else:
            words = query.strip().split()
        return words

    def mapping(self, query, clear = True):
        words = self.preprocess(query, clear)

        #print query, '->', words
        renamed, vectors = self.word2vec.read(words)
        renamed2vec = dict(zip(renamed, vectors))

        if len(renamed) != len(words):
            vectors = []
            for word in words:
                if word in renamed2vec:
                    vectors.append(renamed2vec[word])

        if len(vectors)>0:
            vec = np.array(vectors).mean(axis=0)

            if self.L1_normalize:
                return self.do_L1_norm(vec)
            if self.L2_normalize:
                return self.do_L2_norm(vec)
            return vec
        else:
            return None



# word2vec + average pooling + fliter stop words
class AveWord2VecFilterStop(AveWord2Vec):

    # datafile: the path of pre-trained word2vec data
    def __init__(self, datafile, ndims = 0, L1_normalize = 0, L2_normalize = 0):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)
        self.word2vec = BigFile(datafile)
        if ndims != 0:
            assert self.word2vec.ndims == self.ndims, "feat dimension is not match %d != %d" % (self.word2vec.ndims, self.ndims)
        else:
            self.ndims = self.word2vec.ndims

    def preprocess(self, query, clear):
        if clear:
            words = clean_str_filter_stop(query)
        else:
            words = query.strip().split()
        return words       




# Bag-of-words
class BoW2Vec(Text2Vec):

    # datafile: the path of bag-of-words vocabulary file
    def __init__(self, datafile, ndims = 0, L1_normalize = 0, L2_normalize = 0):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)
        word_vob = map(str.strip, open(datafile).readlines())
        self.word2index = dict(zip(word_vob, range(len(word_vob))))
        if ndims != 0:
            assert len(word_vob) == self.ndims, "feat dimension is not match %d != %d" % (len(word_vob), self.ndims)
        else:
            self.ndims = len(word_vob)
        printStatus(INFO + '.' + self.__class__.__name__, "%d words" % self.ndims)

    def preprocess(self, query):
        return clean_str(query)

    def mapping(self, query):
        words = self.preprocess(query)

        vec = [0.0]*self.ndims
        for word in words:
            if word in self.word2index:
                vec[self.word2index[word]] += 1
            # else:
            #     print word

        if sum(vec) > 0:
            if self.L1_normalize:
                vec = self.do_L1_norm(vec)
            if self.L2_normalize:
                vec = self.do_L2_norm(vec)
            return vec
            
        ###############################
        # sometimes need to modify
        # else:
        #     return None
        ###############################
        else:
            return vec
            



# Bag-of-words + fliter stop words
class BoW2VecFilterStop(BoW2Vec):


    def preprocess(self, query):
        return clean_str_filter_stop(query)






# convert word to index for RNN
class Index2Vec(Text2Vec):
    # datafile: the path of bag-of-words vocabulary file
    def __init__(self, datafile, ndims = 0, L1_normalize = 0, L2_normalize = 0, we_vocab = None):
        Text2Vec.__init__(self, datafile, ndims, L1_normalize, L2_normalize)

        if we_vocab is None:
            word_vob = map(str.strip, open(datafile).readlines())
        else:
            word_vob = we_vocab
        if 'UNK' not in word_vob:
            word_vob.append('UNK')
        # Reserve 0 for masking via pad_sequences
        self.word2index = dict(zip(word_vob, range(1,len(word_vob)+1)))
        self.n_vocab = len(word_vob) + 1
        self.vocab = word_vob

    def preprocess(self, query):
        return clean_str(query)

    def mapping(self, query):
        words = self.preprocess(query)
        vec = []
        for word in words:
            if word in self.word2index:
                idx = self.word2index[word]
            else: 
                idx = self.word2index['UNK']
            vec.append(idx)

        if len(vec) > 0:
            return vec
        else:
            return None







NAME_TO_ENCODER = {'word2vec': AveWord2Vec, 'word2vec_mean': AveWord2Vec, 'word2vec_filterstop': AveWord2VecFilterStop,
                   'bow': BoW2Vec, 'bow_filterstop': BoW2VecFilterStop,
                   'lstm': Index2Vec, 'bilstm': Index2Vec, 'gru': Index2Vec, 'bigru': Index2Vec}


def get_text_encoder(name):
    return NAME_TO_ENCODER[name]



if __name__ == '__main__':
    corpus = 'flickr'
    word2vec_model = 'vec500flickr30m'
    text_data_path = os.path.join(rootpath, "word2vec", corpus, word2vec_model)
    
    collection = 'mscoco2014train'
    encoder_bow = BoW2Vec(os.path.join(rootpath, collection, 'TextData/vocabulary/bow/word_vocab_0.txt'), L1_normalize=1)
    encoder_bowfs = BoW2VecFilterStop(os.path.join(rootpath, collection, 'TextData/vocabulary/bow_filterstop/word_vocab_0.txt'), L1_normalize=1)

   
    encoder_w2vfs = get_text_encoder('word2vec_filterstop')(text_data_path, L1_normalize=1)

    query = "dog is running in boy"

    #for encoder in [encoder_bow, encoder_bowfs, encoder_wh, encoder_whfs, encoder_w2vfs]:
    for encoder in [encoder_bow, encoder_bowfs]:
        feat = encoder.embedding(query)
        print len(feat), feat.min(), feat.max(), sum(feat)
