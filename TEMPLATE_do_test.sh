rootpath=@@@rootpath@@@
trainCollection=@@@trainCollection@@@
valCollection=@@@valCollection@@@
testCollection=@@@testCollection@@@
overwrite=0

n_caption=@@@n_caption@@@

# model info
model_path=@@@model_path@@@
weight_name=@@@weight_name@@@


python w2vv_tester.py  $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite --model_path $model_path --weight_name $weight_name --n_caption $n_caption

