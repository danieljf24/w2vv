ROOTPATH=$HOME/VisualSearch
mkdir -p $ROOTPATH && cd $ROOTPATH

#download and extract dataset
wget http://lixirong.net/data/w2vv-tmm2018/flickr8k.tar.gz
tar zxf flickr8k.tar.gz

wget http://lixirong.net/data/w2vv-tmm2018/flickr30k.tar.gz
tar zxf flickr30k.tar.gz

#download and extract pre-trained word2vec
wget http://lixirong.net/data/w2vv-tmm2018/word2vec.tar.gz
tar zxf word2vec.tar.gz

cd -
