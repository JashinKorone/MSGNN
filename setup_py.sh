echo 'Please setup CUDA 10.1 first'

yes | pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
yes | pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
yes | pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
yes | pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
yes | pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
yes | pip install torch-geometric
yes | pip install tensorboardX
yes | pip install geopy
yes | pip install pandas
yes | pip install pandarallel
yes | pip install numpy
yes | pip install scipy
yes | pip install matplotlib
yes | pip install tqdm