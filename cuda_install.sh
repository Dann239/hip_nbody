curl http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1810/x86_64/cuda-repo-ubuntu1810_10.1.168-1_amd64.deb > cuda_repo.deb
sudo dpkg -i cuda_repo.deb
rm cuda_repo.deb

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1810/x86_64/7fa2af80.pub
sudo apt-get update -y
sudo apt-get install cuda -y

echo 'export PATH=$PATH:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1' | sudo tee -a /etc/profile.d/cuda.sh
echo '/usr/local/cuda-10.1/lib64' | sudo tee -a /etc/ld.so.conf.d/cuda-10-1.conf
sudo ldconfig
