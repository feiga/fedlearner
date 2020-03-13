FROM tensorflow/tensorflow:1.15.2

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN make protobuf

ENV PYTHONPATH=/app:$PYTHONPATH

CMD []
