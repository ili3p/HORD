CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 139 ensemble &
CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 159 ensemble &
CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 539 ensemble 

CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 339 ensemble &
CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 759 ensemble &
CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 589 ensemble 

CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 639 ensemble &
CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 149 ensemble &
CUDA_VISIBLE_DEVICES=$1 python pySOT_Ensemble_runner.py 1 200 529 ensemble 
