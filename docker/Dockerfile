FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip3 install --upgrade pip  # enable PEP 660 support
WORKDIR /app
RUN pip3 install transformers==4.35.0 "redis[hiredis]==5.0.1"