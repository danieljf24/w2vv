trainCollection=$1
valCollection=$2
testCollection=$3
overwrite=0

# Generate a dictionary on the training set
./do_gene_vocab.sh $trainCollection


# training
python w2vv_trainer.py $trainCollection $valCollection  $testCollection --overwrite $overwrite 


# evaluation (Notice that a script file do_test_${testCollection}.sh will be automatically generated when the training process ends.)
./do_test_${testCollection}.sh
