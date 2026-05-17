# pip install torch==2.0.0

pip install -e .
pip install -e vendor/esm


# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu113.html
# conda install -c pyg pytorch-scatter -y
# conda install -c pyg pytorch-sparse -y
# conda install -c pyg pytorch-cluster -y
# conda install -c pyg pytorch-spline-conv -y
pip install torch -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html



pip install torch_geometric biotite
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
