
echo "install torch, torchvision, torchaudio (edit this line for your CUDA version)"
pip install torch==2.3.0+cu121 torchvision==2.3.0+cu121 torchaudio==0.18.0 -f https://download.pytorch.org/whl/torch_stable.html

echo "installing requirements"
cd OvSGTR
pip install -r requirements.txt

echo "installing GroundingDino"
cd GroundingDINO && python3 setup.py install 

echo "creating weights directory"
mkdir -p $PWD/weights

echo "downloading SGG checkpoint"
wget -nc "https://huggingface.co/JosephZ/OvSGTR/resolve/main/vg-ovdr-swinb-mega-best.pth?download=true" -O $PWD/weights/vg-ovdr-swinb-mega-best.pth

echo "downloading VG prompt dicts"
mkdir -p $PWD/../../data
wget -nc "https://huggingface.co/JosephZ/OvSGTR/resolve/main/vg_data.tar.gz" -O $PWD/../../data/vg_data.tar.gz

echo "extracting VG prompt dicts"
tar -xf $PWD/../../data/vg_data.tar.gz -C $PWD/../../data

echo "Done"