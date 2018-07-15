trainCollection=$1
valCollection=$2
testCollection=$3
img_feature=$4
n_caption=$5

overwrite=0

# Generate a dictionary on the training set
./do_gene_vocab.sh $trainCollection


# training
python w2vv_trainer.py $trainCollection $valCollection  $testCollection --overwrite $overwrite --img_feature $img_feature --n_caption $n_caption


# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
./do_test_${testCollection}.sh
