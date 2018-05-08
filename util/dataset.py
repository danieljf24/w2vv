import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences



def read_img2sents_ms(input_file, qry2vec, qry2vec_2, qry2vec_3):
    print 'reading data from:', input_file
    counter = 0
    img2sents = {}
    for line in open(input_file):
        sid, sent = line.strip().split(' ', 1)
        img = sid.split("#")[0].strip().split('.jpg')[0]
        if qry2vec.mapping(sent) is None or qry2vec_2.mapping(sent) is None or qry2vec_3.mapping(sent) is None:
            print sent
            continue
        img2sents.setdefault(img, []).append(sent)
        counter += 1
    return img2sents, counter





class PairDataSet_MS():
    def __init__(self, caption_file, batch_size, qry2vec, bow2vec, w2v2vec, img_feats, flag_maxlen=False, maxlen=32):
        self.img2sents, self.datasize = read_img2sents_ms(caption_file, qry2vec, bow2vec, w2v2vec)
        self.img_list = self.img2sents.keys()
        self.img_feats = img_feats
 
        self.qry2vec = qry2vec
        self.bow2vec = bow2vec
        self.w2v2vec = w2v2vec
 
        self.batch_size = batch_size
        self.max_batch_size = int( np.ceil(1.0 * self.datasize / batch_size))
 
        self.flag_maxlen = flag_maxlen
        self.maxlen = maxlen

 
    def getBatchData(self):
        counter = 0
        while 1:
            img_list = random.sample(self.img_list, self.batch_size)
            query_list = [random.choice(self.img2sents[img]) for img in img_list]
     
     
            text_X = []
            text_X_1 = []
            #renamed, feats = qry_feats.read(query_list)
            for query in query_list:
                text_X.append(self.qry2vec.mapping(query))
                text_X_1.append(list(self.bow2vec.mapping(query)) + list(self.w2v2vec.mapping(query)))
                
            img_X = []
            renamed, feats = self.img_feats.read(img_list)
            for img in img_list:
               img_X.append(feats[renamed.index(img)])
     

            if self.flag_maxlen:
                text_X = pad_sequences(text_X, maxlen=self.maxlen, truncating='post')

            # X=query Y=image
            yield (np.array(img_X), [np.array(text_X), np.array(text_X_1)])