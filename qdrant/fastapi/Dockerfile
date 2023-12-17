FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime


RUN mkdir /fastapi
WORKDIR /fastapi
COPY . /fastapi

RUN apt-get update -y && \
    apt-get install -y python3-pip
RUN pip install -r requirements.txt

EXPOSE 7861

CMD ["python", "-m", "app"]