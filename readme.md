# Install Requirements
```sh
pip install -r requirements.txt
```
We recommended use cuda 11.2 and cuDNN 8.1.0

## Install PyTorch Gpu
```sh
pip install torch==2.1.0+cu111 torchvision==0.16.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

if you got importError: cannot import name 'builder' from 'google.protobuf.internal'
the <code>builder.py</code> file is missing from the protobuf package created from TensorFlow. A workaround is to simply copy the latest copy of <code>builder.py</code> from the [protobuf repository](https://raw.githubusercontent.com/protocolbuffers/protobuf/main/python/google/protobuf/internal/builder.py) into your local drive

copied and pasted this code and made it into a .py then took that file and pasted it in {your env location}/site-packages/google/protobuf/internal/builder.py