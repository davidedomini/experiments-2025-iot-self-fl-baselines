FROM python:3.12
RUN mkdir -p /experiment
VOLUME "davide.domini-volume"
ENV DATA_DIR=/data
WORKDIR /experiment
COPY requirements.txt /experiment
RUN python3 -m pip install -r requirements.txt
COPY . /experiment
ENV OWNER=1000:1000
CMD export OUTPUT_DIR=$DATA_DIR/$(date +%Y-%m-%d-%H-%M-%S)-$(hostname) && \
    mkdir -p $OUTPUT_DIR && \
    python3 src/main.py | tee $OUTPUT_DIR/output.log && \
    chown -R $OWNER $DATA_DIR