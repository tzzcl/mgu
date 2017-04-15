THEANO_FLAGS='floatX=float32,device=gpu0,nvcc.fastmath=True' python IRNN_gru_2014.py > ./result/2014_100.csv &
THEANO_FLAGS='floatX=float32,device=gpu1,nvcc.fastmath=True' python IRNN_gru_2015.py > ./result/2015_100.csv &
