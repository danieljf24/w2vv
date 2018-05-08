import re
import numpy as np
# from nltk.corpus import stopwords
from simpleknn.bigfile import BigFile
from keras.preprocessing.sequence import pad_sequences

# ENGLISH_STOP_WORDS = stopwords.words('english')
ENGLISH_STOP_WORDS = "i me my myself we our ours ourselves you your yours yourself yourselves he him his himself she her hers herself it its itself they them their theirs themselves what which who whom this that these those am is are was were be been being have has had having do does did doing a an the and but if or because as until while of at by for with about against between into through during before after above below to from up down in out on off over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very s t can will just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan shouldn wasn weren won wouldn".strip().split()


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()


def clean_str_filter_stop(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    cleaned_string = string.strip().lower().split()
    # remove stop words
    return [word for word in cleaned_string if word not in ENGLISH_STOP_WORDS]


def get_we_parameter(vocabulary, word2vec_file):
    print 'getting inital word embedding ...'
    w2v_reader = BigFile(word2vec_file)
    ndims = w2v_reader.ndims    
    we = []
    # Reserve 0 for masking via pad_sequences
    we.append([0]*ndims)
    fail_counter = 0
    for word in vocabulary:
        word = word.strip()
        try:
            vec = w2v_reader.read_one(word)
            # print vec
            we.append(vec)
        except Exception, e:
            # print word
            vec = np.random.uniform(-1,1,ndims)
            we.append(vec)
            fail_counter +=1 
    print "%d words out of %d words cannot find pre-trained word2vec vector" % (fail_counter, len(vocabulary))
    return np.array(we)

def encode_text(opt,text2vec,bow2vec,w2v2vec,sent):
    sent_vec = text2vec.mapping(sent)
    rnn_vec = pad_sequences([sent_vec], maxlen=opt.sent_maxlen,  truncating='post')[0]
    sent_bow_vec = bow2vec.mapping(sent)
    sent_w2v_vec = w2v2vec.mapping(sent)
    return rnn_vec, list(sent_bow_vec)+list(sent_w2v_vec)


if __name__ == '__main__':
    test_strs = '''a Dog is running
        The dog runs
        dogs-x runs'''.split('\n')

    for t in test_strs:
        print t, '->', clean_str(t), '->', clean_str_filter_stop(t)
        print generate_letter_grams(t)

