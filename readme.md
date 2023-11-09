# Install Requirements
```sh
pip install -r requirements.txt
```
We recommended use cuda 11.8 and cuDNN 8.6.0

## Cloning yolov5 submodule
```sh  
git submodule update --init
```

## Install PyTorch GPU
```sh
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

if you got importError: cannot import name 'builder' from 'google.protobuf.internal'
the <code>builder.py</code> file is missing from the protobuf package created from TensorFlow. A workaround is to simply copy the latest copy of <code>builder.py</code> from the [protobuf repository](https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py) into your local drive

copied and pasted this code and made it into a .py then took that file and pasted it in {your env location}/site-packages/google/protobuf/internal/builder.py