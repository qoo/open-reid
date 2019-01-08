# 1. KEEP UBUNTU OR DEBIAN UP TO DATE

sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade
sudo apt-get -y autoremove

# install utilities 
sudo apt-get install tmux -y
sudo apt-get install htop -y

# 2. INSTALL THE DEPENDENCIES

# Build tools:
#sudo apt-get install -y build-essential cmake

# GUI (if you want to use GTK instead of Qt, replace 'qt5-default' with 'libgtkglext1-dev' and remove '-DWITH_QT=ON' option in CMake):
#sudo apt-get install -y qt5-default libvtk6-dev

# INSTALL THE git
sudo apt install git-all -y

# INSTALL python for tensorflow
sudo apt -y install python-dev python-pip

# cudnn for cuda 9.0  ubuntu
wget -O cudnn-9.0-linux-x64-v7.3.0.29.tar https://www.dropbox.com/s/ezlfz6xccpydkbu/cudnn-9.0-linux-x64-v7.3.0.29.tar?dl=0

tar -xvf cudnn-9.0-linux-x64-v7.3.0.29.tar

sudo cp -P cuda/lib64/* /usr/local/cuda/lib64/

sudo cp -P cuda/include/* /usr/local/cuda/include/

sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
#check CuDNN version
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2

# check Cuda version
cat /usr/local/cuda/version.txt

pip install tf-nightly-gpu==1.13.0-dev20181203


#### torch
#python 3.5
# pip3 install torch torchvision
# python 2.7
# pip install torch torchvision

#pip install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl
#pip uninstall https://download.pytorch.org/whl/cu90/torch-0.4.1-cp27-cp27mu-linux_x86_64.whl

pip install https://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
pip install torchvision
pip install scikit-learn
pip install metric_learn
pip install Pillow

# python 3 libary
#sudo apt install python3-pip -y
#pip3 install scikit-learn
#pip3 install metric_learn
#pip3 install h5py

#git clone git@github.com:qoo/open-reid.git
git clone https://github.com/qoo/open-reid.git

export PYTHONPATH=~/open-reid

# download datasets
# wget -O /root/open-reid/examples/data/cuhk03/raw/cuhk03_release.zip https://www.dropbox.com/s/ezlfz6xccpydkbu/cudnn-9.0-linux-x64-v7.3.0.29.tar?dl=0 https://www.dropbox.com/s/g63s3lix7x6p5ln/cuhk03_release.zip?dl=0
wget -O orb_sample_result_train_all_160.zip https://www.dropbox.com/s/64thsa9f885l8hm/orb_sample_result_train_all_160.zip?dl=0
unzip orb_sample_result_train_all_160.zip

