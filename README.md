1.合并数据
cd /home/utopia/CVDATA/German_AI_Challenge_2018/session1
cat training.h5.z01 training.h5.z02 training.h5.z03 training.h5.z04 training.h5.zip > training.full.zip
unzip training.full.zip （sha512sum training.h5 ok）
unzip validation.h5.zip （√）
unzip round1_test_a_20181109.zip （√）

2.安装相关工具，无GPU请安装CPU版本
pip3 install -U scikit-learn (可选)
git clone https://github.com/tensorflow/models.git (可选)
pip3 install ~/software/tensorflow_gpu-1.9.0-cp36-cp36m-manylinux1_x86_64.whl (可选)
pip3 install tflearn (可选，ResNeXt使用，建议使用此网络)
pip3 install ~/software/xgboost-0.81-py2.py3-none-manylinux1_x86_64.whl (可选)
pip3 install ~/software/torch-0.4.1-cp36-cp36m-linux_x86_64.whl (可选)
pip3 install torchvision (可选)
pip3 install keras (可选，NASNet使用)

3.查看tensorboard
~/.local/bin/tensorboard --logdir='/tmp/tflearn_logs'

4.入门资源
https://tianchi.aliyun.com/notebook/detail.html?spm=5176.8366600.0.0.1e6b311fIYlDNY&id=36602
http://tianchi-tum.oss-eu-central-1.aliyuncs.com/analysis.ipynb
