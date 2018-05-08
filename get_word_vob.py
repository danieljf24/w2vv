# 
#  We convert all sentences to lowercase, discard non-alphanumeric characters. 
#  We filter words to those that occur at least $freq_threshold times in the training set.
#  Ref: Deep Visual-Semantic Alignments for Generating Image Descriptions  (Data Preprocessing)
# 
import os
import sys


from basic.constant import ROOT_PATH as rootpath
from basic.common import printMessage, checkToSkip, makedirsforfile
from util.text import clean_str, clean_str_filter_stop


INFO = __file__

if __name__ == "__main__":

    if len(sys.argv)  < 5:
        print "usage: python %s collection text_style freq_threshold overwrite" % INFO
        sys.exit(0)

    collection_list = [sys.argv[1]]
    text_style = sys.argv[2]         # "bow" | "bow_filterstop"
    freq_threshold = int(sys.argv[3])
    overwrite = int(sys.argv[4])


    for collection in collection_list:
        print "processing %s ..." % collection

        input_file = os.path.join( rootpath, "%s/TextData/%s.caption.txt" % (collection, collection) )
        output_vocab_file = os.path.join( rootpath, "%s/TextData/vocabulary/%s/word_vocab_%d.txt" % (collection, text_style, freq_threshold) )
        output_vocab_counter_file = os.path.join( rootpath, "%s/TextData/vocabulary/%s/word_vocab_counter_%d.txt" % (collection, text_style, freq_threshold) )


        if checkToSkip(output_vocab_file, overwrite):
            sys.exit(0)
        if checkToSkip(output_vocab_counter_file, overwrite):
            sys.exit(0)
        makedirsforfile(output_vocab_file)


        word2counter = {}
        len2counter ={}
        for index, line in enumerate(open(input_file)):
            sid, sent = line.strip().split(" ", 1)
            if text_style == "bow":
                sent = clean_str(sent)
            elif text_style == "bow_filterstop":
                sent = clean_str_filter_stop(sent)
            length = len(sent)
            len2counter[length] = len2counter.get(length, 0) + 1
            if index == 0:
                print line.strip()
                print 'After processing: ', sid, sent
                print '\n'
            for word in sent:
                word2counter[word] = word2counter.get(word, 0) + 1

        sorted_wordCounter = sorted(word2counter.iteritems(), key = lambda a:a[1], reverse=True)


        output_line_vocab = [ x[0] for x in sorted_wordCounter if x[1] >= freq_threshold ]
        output_line_vocab_counter = [ x[0] + ' '  + str(x[1]) for x in sorted_wordCounter if x[1] >= freq_threshold ]

        open(output_vocab_file, 'w').write('\n'.join(output_line_vocab))
        open(output_vocab_counter_file, 'w').write('\n'.join(output_line_vocab_counter))

        print sorted(len2counter.items(), key = lambda x:x[0], reverse=True)
