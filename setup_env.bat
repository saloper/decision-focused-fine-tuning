@echo off
call conda create -n dffttimes python=3.11 -y
call conda activate dffttimes
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install ..\wheels\diffcp-1.1.8-cp311-cp311-win_amd64.whl
pip install -e .[times]
pip install jupyter matplotlib pyarrow tensorboard