import os
import sys
import time
import json
import random
import numpy as np

from basic.constant import *
from basic.metric import getScorer
from basic.common import makedirsforfile, checkToSkip, writePkl

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

from keras.utils import generic_utils
import tensorboard_logger as tb_logger

from w2vv import get_model
from w2vv_pred import pred_mutual_error_ms

from simpleknn.bigfile import BigFile
from util.dataset import PairDataSet_MS
from util.losser import get_losser
from util.text import get_we_parameter
from util.text2vec import get_text_encoder
from util.evaluation import i2t_map
from util.util import readImgSents


INFO = __file__


def cal_val_perf(all_errors, opt=None):
    
    # validation metric: MAP
    i2t_map_score = 0
    i2t_map_score = i2t_map(all_errors, n_caption=opt.n_caption)
    
    currscore = i2t_map_score

    return  currscore



def process(opt, trainCollection, valCollection, testCollection):

    rootpath = opt.rootpath
    overwrite =  opt.overwrite

    opt.n_text_layers = map(int, opt.n_text_layers.strip().split('-'))

    if opt.init_model_from != '':
        assert opt.img_feature in opt.init_model_from
        init_model_name = opt.init_model_from.strip().split("/")[-1]
        train_style = opt.model_name + "_" +  INFO + "_ft_" + init_model_name
    else:
        train_style = opt.model_name + "_" +  INFO

    
    # text embedding style
    if '@' in opt.text_style and opt.model_name.endswith('_ms'):
        rnn_style, bow_style, w2v_style = opt.text_style.strip().split('@')
        opt.rnn_style = rnn_style
        text_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", "bow", opt.rnn_vocab)
        bow_data_path = os.path.join(rootpath, trainCollection, "TextData", "vocabulary", bow_style, opt.bow_vocab)
        w2v_data_path = os.path.join(rootpath, "word2vec", opt.corpus,  opt.word2vec)
        text_name = opt.bow_vocab + "_rnn_%d_%s_sent_%d" % (opt.rnn_size, opt.rnn_vocab, opt.sent_maxlen)
    else:
        print opt.text_style + " is not supported, please check the 'text_style' parameter"
        sys.exit(0)

    optm_style = opt.optimizer + '_clipnorm_%.1f_lr_%.5f_dp_%.2f_l2_%.5f_%s_bs_%d' % \
        (opt.clipnorm, opt.lr, opt.dropout, opt.l2_p, opt.loss_fun, opt.batch_size)
    model_style =  "-".join(map(str, opt.n_text_layers)) + '_' + opt.hidden_act + '_' + opt.simi_fun

    checkpoint_dir = os.path.join(rootpath, trainCollection, opt.checkpoint, 'w2vv', valCollection, 
        train_style, opt.text_style + '_' + text_name, opt.img_feature, optm_style, model_style, opt.postfix)

    # output visualization script
    runfile_vis = 'do_visual.sh'
    open( runfile_vis, 'w' ).write('port=$1\ntensorboard --logdir %s --port $port' % checkpoint_dir)
    os.system('chmod +x %s' % runfile_vis)


    val_per_hist_file = os.path.join(checkpoint_dir, 'val_per_hist.txt')
    if checkToSkip(val_per_hist_file, overwrite):
        sys.exit(0)
    # else:
    #     if os.path.exists(checkpoint_dir):
    #         os.system("rm -r " + checkpoint_dir)
    makedirsforfile(val_per_hist_file)
    model_file_name = os.path.join(checkpoint_dir, 'model.json')
    model_img_name = os.path.join(checkpoint_dir, 'model.png')
    tb_logger.configure(checkpoint_dir, flush_secs=5)


    # text embedding (text representation)
    if '@' in opt.text_style and opt.model_name.endswith('_ms'):
        text2vec = get_text_encoder(rnn_style)(text_data_path)
        bow2vec = get_text_encoder(bow_style)(bow_data_path)
        w2v2vec = get_text_encoder(w2v_style)(w2v_data_path)
        if opt.n_text_layers[0] == 0:
            opt.n_text_layers[0] = bow2vec.ndims + w2v2vec.ndims
        else:
            assert opt.n_text_layers[0] == bow2vec.ndims + w2v2vec.ndims
        opt.vocab_size = text2vec.n_vocab
        opt.embed_size = w2v2vec.ndims
    else:
        text2vec = get_text_encoder(opt.text_style)(text_data_path, ndims=opt.n_text_layers[0])    
        if opt.n_text_layers[0] == 0:
            opt.n_text_layers[0] = text2vec.ndims
        
    # img2vec
    img_feat_path = os.path.join(rootpath, trainCollection, 'FeatureData', opt.img_feature)
    img_feats = BigFile(img_feat_path)

    val_img_feat_path = os.path.join(rootpath, valCollection, 'FeatureData', opt.img_feature)
    val_img_feats = BigFile(val_img_feat_path)

    # write out options for evaluation    
    pkl_file = os.path.join(checkpoint_dir, 'option.pkl')
    writePkl(opt, pkl_file)


    # define word2visualvec model     
    if opt.model_name.endswith('_ms'):
        we_weights = get_we_parameter(text2vec.vocab, w2v_data_path)
        print we_weights.shape
        model = get_model(opt.model_name)(opt, we_weights=we_weights)
    else:
        model = get_model(opt.model_name)(opt)
    model.save_json_model(model_file_name)
    model.plot(model_img_name)
    model.compile_model(opt.loss_fun, opt=opt)
    if opt.init_model_from != '':
        print '*'*20
        print 'initialize the model form ' + opt.init_model_from
        print '*'*20
        model.init_model(opt.init_model_from)


    # training set
    caption_file = os.path.join(rootpath, trainCollection, 'TextData', '%s.caption.txt' % trainCollection)
    trainData = PairDataSet_MS(caption_file, opt.batch_size, text2vec, bow2vec, w2v2vec, img_feats, flag_maxlen=True, maxlen=opt.sent_maxlen)


    val_sent_file = os.path.join(rootpath, valCollection, 'TextData', '%s.caption.txt' % valCollection)
    val_img_list, val_sents_id, val_sents = readImgSents(val_sent_file)

    
    losser = get_losser(opt.simi_fun)()


    best_validation_perf = 0
    n_step = 0
    count = 0
    lr_count = 0
    best_epoch = -1
    val_per_hist = []
    for epoch in range(opt.max_epochs):
        print '\nEpoch', epoch
        print "Training..."
        print "learning rate: ", model.get_lr()
        tb_logger.log_value('lr', model.get_lr(), step=n_step)

        train_progbar = generic_utils.Progbar(trainData.datasize)
        trainBatchIter = trainData.getBatchData()
        for minibatch_index in xrange(trainData.max_batch_size):
        # for minibatch_index in xrange(10):
            n_step += 1
            img_X_batch, text_X_batch = trainBatchIter.next()
            loss_batch = model.model.train_on_batch(text_X_batch, img_X_batch)
            train_progbar.add(img_X_batch.shape[0], values=[("loss", loss_batch)])

            tb_logger.log_value('loss', loss_batch, step=n_step)
            tb_logger.log_value('n_step', n_step, step=n_step)

        print "\nValidating..."
        all_errors = pred_mutual_error_ms(val_img_list, val_sents, model, text2vec, bow2vec, w2v2vec, val_img_feats, losser, opt=opt)
        
        this_validation_perf = cal_val_perf(all_errors, opt=opt)
        tb_logger.log_value('val_accuracy', this_validation_perf, step=n_step)
        
        val_per_hist.append(this_validation_perf)

        print 'previous_best_performance: %.3f' % best_validation_perf
        print 'current_performance: %.3f' % this_validation_perf

        fout_file = os.path.join(checkpoint_dir, 'epoch_%d.h5' % ( epoch))

        lr_count += 1
        if this_validation_perf > best_validation_perf:
            best_validation_perf = this_validation_perf          
            count = 0
            
            # save model
            model.model.save_weights(fout_file)
            if best_epoch != -1:
                os.system('rm '+ os.path.join(checkpoint_dir, 'epoch_%d.h5' % ( best_epoch)))
            best_epoch = epoch

        else:
            # when the validation performance has decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_count > 2:
                model.decay_lr(0.5)
                lr_count = 0
            count += 1
            if count > 10:
                print ("Early stopping happened")
                break


    sorted_epoch_perf = sorted(zip(range(len(val_per_hist)), val_per_hist), key = lambda x: x[1], reverse=True)
    with open(val_per_hist_file, 'w') as fout:
        for i, perf in sorted_epoch_perf:
            fout.write("epoch_" + str(i) + " " + str(perf) + "\n")


    # generate the shell script for test
    templete = ''.join(open( 'TEMPLATE_do_test.sh'  ).readlines())
    striptStr = templete.replace('@@@rootpath@@@', rootpath)
    striptStr = striptStr.replace('@@@trainCollection@@@', trainCollection)
    striptStr = striptStr.replace('@@@valCollection@@@', valCollection)
    striptStr = striptStr.replace('@@@testCollection@@@', testCollection)
    striptStr = striptStr.replace('@@@model_path@@@', checkpoint_dir)
    striptStr = striptStr.replace('@@@weight_name@@@', 'epoch_%d.h5' % sorted_epoch_perf[0][0])
    striptStr = striptStr.replace('@@@n_caption@@@', str(opt.n_caption))

    print os.path.join(checkpoint_dir, 'epoch_%d.h5' % sorted_epoch_perf[0][0])
    runfile = 'do_test_%s.sh' % (testCollection)
    open( runfile, 'w' ).write(striptStr+'\n')
    os.system('chmod +x %s' % runfile)
    # os.system('./'+runfile)
    os.system('cp %s/epoch_%d.h5 %s/best_model.h5' % (checkpoint_dir, sorted_epoch_perf[0][0], checkpoint_dir))


    

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [opt] trainCollection valCollection testCollection""")

    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)

    parser.add_option('--checkpoint', type="string", default='cv_w2vv_new', help='output directory to write checkpoints to (default: cv_w2vv_new)')
    parser.add_option('--init_model_from', type="string", default='', help='initialize the model parameters from some specific checkpoint?')
    parser.add_option("--model_name", default='w2vv_ms', type="string", help="model name (default: w2vv_ms)")

    # image feature
    parser.add_option("--img_feature", default=DEFAULT_IMG_FEATURE, type="string", help="image feature (default: %s)"%DEFAULT_IMG_FEATURE)

    # text embedding
    parser.add_option("--text_style", default=DEFAULT_TEXT_STYLE, type="string", help="text embedding style (default: %s)"%DEFAULT_TEXT_STYLE)
    # word2vector parameters
    parser.add_option("--corpus", default=DEFAULT_CORPUS, type="string", help="corpus using which word2vec was trained (default: %s)" % DEFAULT_CORPUS)
    parser.add_option("--word2vec", default=DEFAULT_WORD2VEC, type="string", help="word2vec model (default: %s)" % DEFAULT_WORD2VEC)
    # bag-of-words parameters
    parser.add_option("--bow_vocab", default=DEFAULT_BOW_VOCAB, type="string", help="bag-of-words vocabulary file name (default: %s)"%DEFAULT_BOW_VOCAB)
    parser.add_option("--rnn_vocab", default=DEFAULT_RNN_VOCAB, type="string", help="rnn vocabulary file name(default: %s)"%DEFAULT_RNN_VOCAB)
    parser.add_option('--rnn_size', type="int", default=1024, help='size of the rnn (default: 1024)')
    parser.add_option("--sent_maxlen", default=32, type="int", help="maximun length of sentece")

    # model parameters
    parser.add_option('--n_text_layers', type="str", default='0-2048-2048', help='number of neurons in each layers (default: 0-2048-2048')  
    parser.add_option('--hidden_act', type="str", default='relu', help='activation function on hidden layer (default: relu)')
    parser.add_option('--dropout', type="float", default=0.2, help='dropout (default: 0.2)')    

    # optimization parameters
    parser.add_option('--max_epochs', type="int", default=100, help='number of epochs to train (default: 100)')
    parser.add_option('--batch_size', type="int", default=100, help='batch size (default: 100)')
    parser.add_option('--optimizer', type="string", default='rmsprop', help='optimization algorithm (default: rmsprop)')
    parser.add_option('--clipnorm', type="float", default='5.', help='l2 W_regularizer (default: 0)')
    parser.add_option('--loss_fun', type="string", default='mse', help='loss function (default: mse)')
    parser.add_option("--simi_fun", default='cosine', type="string", help="similarity function: dot cosine")

    parser.add_option('--lr', type="float", default=0.0001, help='learning rate (default: 0.001)')
    parser.add_option('--l2_p', type="float", default='0.0', help='l2 W_regularizer (default: 0)')

    parser.add_option('--n_caption', type="int", default='5', help='number of captions of each image/video (default: 5)')
    parser.add_option('--postfix', type="str", default='run_0', help='')

    
    (opt, args) = parser.parse_args(argv)
    if len(args) < 3:
        parser.print_help()
        return 1

    print json.dumps(vars(opt), indent = 2)
    return process(opt, args[0], args[1], args[2])


if __name__ == "__main__":
    sys.exit(main())
