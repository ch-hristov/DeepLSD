# Specs:
# Homographies : A single .zip file of a folder containing hdf5 files with homographies.

pip install scikit-build
pip install -r requirements.txt  # Install the requirements
cd third_party/progressive-x/graph-cut-ransac/build; cmake ..; make -j8; cd ../../../..  # Install the C++ library Graph Cut RANSAC
cd third_party/progressive-x/build; cmake ..; make -j8; cd ../../..  # Install the C++ library Progressive-X
pip install -e third_party/progressive-x  # Install the Python bindings of Progressive-X for VP estimation
cd third_party/afm_lib/afm_op; python setup.py build_ext --inplace; rm -rf build; cd ..; pip install -e .; cd ../..  # Install the Cuda code to generate AFM from lines (taken from https://github.com/cherubicXN/afm_cvpr2019)
pip install -e line_refinement  # Install the Python bindings to optimize lines wrt a distance/angle field
pip install -e third_party/homography_est  # Install the code for homography estimation from lines
pip install -e third_party/pytlbd  # Install the LBD line matcher for evaluation
pip install -e .  # Install DeepLSD

mkdir data
mkdir experiments

# Download data
python3 download.py

mkdir weights
wget https://www.polybox.ethz.ch/index.php/s/FQWGkH57UNTqlJZ/download -O weights/deeplsd_wireframe.tar
wget https://www.polybox.ethz.ch/index.php/s/XVb30sUyuJttFys/download -O weights/deeplsd_md.tar

mv ./weights/deeplsd_wireframe.tar ./experiments/deeplsd-d8n-dp/checkpoint_0_0.tar

# Unzip images & labels
unzip dataset.zip

mv ./lines ./engisense-lines
mv ./engisense-lines ./data

# Homographies dir
mkdir ./data/engisense-lines/homographies

# Unzip homogprahies

unzip homographies.zip
mv .data//homographies ./train
mv ./train ./data/engisense-lines/homographies

# Dataset config
mv ./data/engisense-lines/images ./data/engisense-lines/train

rm -r -f dataset.zip
rm -r -f homographies.zip
wandb login $WANDB_LOGIN