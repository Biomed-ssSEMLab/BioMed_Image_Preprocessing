# Introduction
A Multi-beam affine stitching and alignment tool for large-scale EM images. It's based on mb_aligner(https://github.com/Gilhirith/mb_aligner).  
Uses python 3.4+.

# Installation
sudo apt-get update  
sudo apt-get upgrade -y  
sudo apt-get install -y build-essential apt-utils  
sudo apt-get install -y git cmake python3-pip checkinstall pkg-config yasm  
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev  
sudo apt-get install -y libxine2-dev libv4l-dev qt5-default libgtk2.0-dev libtbb-dev libatlas-base-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev v4l-utils libhdf5-dev  

#== opencv ==#  
cd ~  
mkdir -p Software/opencv && cd Software/opencv  
mkdir tmp && cd tmp  
git clone https://github.com/jinhaiqun/opencv-3.1.0.git  
mv opencv-3.1.0/* ../  
cd .. && rm -rf tmp  
mkdir opencv_install-3.1.0  
cd opencv-3.1.0 && mkdir build && cd build  
cmake -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.1.0/modules -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../../opencv_install-3.1.0 -D INSTALL_C_EXAMPLES=OFF -D BUILD_EXAMPLES=ON -D ENABLE_PRECOMPILED_HEADERS=OFF -D WITH_TBB=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_opencv_python3=OFF ..  
make -j8  
make install  
cd ../3rdparty/ippicv/downloads/linux-808b791a6eac9ed78d32a7666804320e/  
tar -zxvf ippicv_linux_20151201.tgz  
sudo cp ippicv_lnx/lib/intel64/libippicv.a /usr/local/lib  
rm -rf ippicv_lnx  
cd ~  
cd Software/opencv  
sudo cp opencv_install-3.1.0/lib/pkgconfig/opencv.pc /usr/lib/pkgconfig/  
sudo echo "~/Software/opencv/opencv_install-3.1.0/lib" | sudo tee /etc/ld.so.conf.d/opencv.conf  
sudo ldconfig  

#== mb_project ==#  
cd ~  
mkdir Projects && cd Projects  
git clone https://github.com/jinhaiqun/mb_project.git  
cd mb_project  
sudo apt-get install -y python3.8-venv  
python3 -m venv venv/  
. venv/bin/activate  
cd 3rd/rh_img_access_layer  
#== (for China) if download too slow, you can use pip --no-cache-dir install -i https://pypi.tuna.tsinghua.edu.cn/simple ==#  
pip --no-cache-dir install opencv-python numpy scipy scikit-learn==0.24.1 Cython protobuf==3.15.0 pillow==8.3.2 xlwt xlrd PyYAML==5.3.1 tqdm  
pip --no-cache-dir install -e .  
cd ../gcsfs  
pip --no-cache-dir install -e .  
cd ../tinyr  
pip --no-cache-dir install -e .  
cd ../rh_config  
pip --no-cache-dir install -e .  
cd ../rh_logger  
pip --no-cache-dir install -e .  
cd ../rh_renderer  
pip --no-cache-dir install -e .  
cd ../..  
pip --no-cache-dir install -e .  

#== cleanup ==#  
sudo apt-get clean autoremove  
sudo rm -rf /var/lib/apt/lists/* /tmp/*  

# Run
cd ~/Projects/mb_project  
. venv/bin/activate  
#== stitching ==#  
python3 mb_aligner/stitching/stitcher.py -i $DATA_DIR -o $OUT_DIR  
#== alignment ==#  
python3 mb_aligner/alignment/aligner.py -i $JSON_DIR -o $OUT_DIR  

 
