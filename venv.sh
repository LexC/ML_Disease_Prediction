sudo apt-get install libicu-dev pkg-config
source /home/lexcc/miniconda3/bin/activate
conda create -n mldp_venv python=3.12
conda activate mldp_venv
pip install ipykernel
pip install -r requirements.txt
