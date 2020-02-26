FROM python:3.8

WORKDIR /usr/src/app

RUN pip install --no-cache-dir --pre guildai

COPY pytest.ini ./
COPY requirements.txt ./
COPY scripts/ ./
COPY setup.* ./

RUN guild init --yes
RUN echo "source guild-env" >> ${HOME}/.bashrc

COPY workflow ./

ENTRYPOINT [ "bash" ]
