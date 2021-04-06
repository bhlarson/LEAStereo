FROM nvcr.io/nvidia/pytorch:21.03-py3
LABEL maintainer="Brad Larson"

RUN echo 'alias py=python' >> ~/.bashrc
RUN apt-get update

# RUN apt-get install -y libsm6 libxext6 ffmpeg # required by opencv-python==4.4.0.42
RUN apt-get install -y libgl1-mesa-glx # for opencv-python


RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
        opencv-python==4.5.1.48 \
        minio==7.0.2 \
        tqdm==4.56.0 \
        natsort==7.0.1 \
        ptvsd==4.3.2 \
        path \
        matplotlib\
        torch \
        torchvision \
        torch \
        tensorboard \
        tensorboardX \
        scipy \
        scikit-image \
        apex

# Tutorial dependencies
RUN pip3 --no-cache-dir install \
        cython \
        pycocotools       

RUN echo 'alias py=python' >> ~/.bashrc

#COPY segment /app/segment/
#COPY utils /app/utils/

RUN git clone https://github.com/NVIDIA/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
# RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /app
ENV LANG C.UTF-8
# port 6006 exposes tensorboard
EXPOSE 6006 
# port 3000 exposes debugger
EXPOSE 3000

# Launch training
#ENTRYPOINT ["python", "segment/train.py"]
RUN ["/bin/bash"]
