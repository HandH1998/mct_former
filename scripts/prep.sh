git clone https://github.com/HandH1998/mct_former.git
cd mct_former
pip install -r requirements.txt
mkdir datasets
cd datasets

hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/datasets/coco.tar.gz && \
tar -xzvf coco.tar.gz && \
rm -rf coco.tar.gz

cd ..

mkdir models
cd models
hdfs dfs -get /home/byte_arnold_hl_mlnlc/user/zhangying.1998/models/coco
cd ..
