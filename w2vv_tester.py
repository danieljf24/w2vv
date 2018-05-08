import os
import re
import sys
import json
import numpy as np
import cPickle as pkl

from basic.constant import *
from basic.common import makedirsforfile, checkToSkip, readPkl, writePkl

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.utils import generic_utils
from w2vv_pred import pred_mutual_error_ms, W2VV_MS_pred

from simpleknn.bigfile import BigFile
from util.losser import get_losser
from util.text2vec  import get_text_encoder
from util.evaluation import i2t
from util.util import readImgSents 


def process(option, trainCollection, valCollection, testCollection):
    
    rootpath = option.rootpath
    overwrite = option.overwrite

    opt_pkl = os.path.join(option.model_path, 'option.pkl')
    opt = readPkl(opt_pkl)
    opt.n_caption = option.n_caption
    opt.model_path = option.model_path
    opt.weight_name = option.weight_name
    
    # result file info
    assert trainCollection in option.model_path
    assert valCollection in option.model_path

    model_path_1, model_path_2 = option.model_path.strip().split('/'+trainCollection+'/')
    model_path = os.path.join(model_path_1, testCollection, 'results', trainCollection, model_path_2)
    model_path = model_path.replace('/%s/'%opt.checkpoint, '/')
    output_dir = os.path.join(model_path, option.weight_name)
    
    result_perf = os.path.join(output_dir, 'perf.txt')
    result_pkl = os.path.join(output_dir, 'test_errors.pkl')
    if checkToSkip(result_perf, overwrite):
        sys.exit(0)
    makedirsforfile(result_perf)

    # text style
    if '@' in opt.text_style and opt.model_name.endswith('_ms'):
        rnn_style, bow_style, w2v_style = opt.text_style.strip().split('@')
        print rnn_style
        text_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", "bow", opt.rnn_vocab)
        bow_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", bow_style, opt.bow_vocab)
        w2v_data_path = os.path.join(rootpath, "word2vec", opt.corpus,  opt.word2vec)
    else:
        print opt.text_style + " is not supported, please check the 'text_style' parameter"
        sys.exit(0)

    # text embedding (text representation)
    text2vec = get_text_encoder(rnn_style)(text_data_path)
    bow2vec = get_text_encoder(bow_style)(bow_data_path)
    w2v2vec = get_text_encoder(w2v_style)(w2v_data_path)


    # img2vec
    img_feats_path = os.path.join(rootpath, testCollection, 'FeatureData', opt.img_feature)
    img_feats = BigFile(img_feats_path)

    # similarity function
    losser = get_losser(opt.simi_fun)()

    # model_name selection
    abs_model_path = os.path.join(opt.model_path, 'model.json')
    weight_path = os.path.join(opt.model_path, opt.weight_name)
    predictor = W2VV_MS_pred(abs_model_path, weight_path)

    test_sent_file = os.path.join(rootpath, testCollection, 'TextData','%s.caption.txt' % testCollection)
    img_list, sents_id, sents = readImgSents(test_sent_file)
    all_errors = pred_mutual_error_ms(img_list, sents, predictor, text2vec, bow2vec, w2v2vec, img_feats, losser, opt=opt)


    # compute performance
    (r1i, r5i, r10i, medri, meanri) = i2t(all_errors, n_caption=opt.n_caption)
    print "Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri, meanri)
    fout_perf = open(os.path.join(output_dir, 'perf.txt'), 'w')
    fout_perf.write("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f\n" % (r1i, r5i, r10i, medri, meanri))
    fout_perf.close()


    writePkl({'errors':all_errors}, result_pkl)





def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [opt] trainCollection valCollection testCollection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    # trained models
    parser.add_option("--model_path", default='cv_keras/pairs/wordhashing/w2vv_clickture_mse_layer_n.py/mincc_3_maximg_30/ruccaffefc7.imagenet/letter_ngram_vocab_50.txt_L1_0_L2_0/rmsprop_lr_0.001_relu_mse_l2_0.00000_dropout_0.200_layer_4_12107-1000-2000-4096', type="string", help="model path")
    parser.add_option("--weight_name", default='epoch_100.h5', type="string", help="weight name")
    
    parser.add_option('--n_caption', type="int", default='5', help='number of captions of each image/video (default: 5)')
    

    (opt, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1

    print json.dumps(vars(opt), indent = 2)
    return process(opt, args[0], args[1], args[2])


if __name__ == "__main__":
    sys.exit(main())    
