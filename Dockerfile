FROM python:3.8

WORKDIR /usr/src/ml-workflow

RUN pip install --no-cache-dir --pre guildai

COPY pytest.ini ./
COPY requirements.txt ./
COPY scripts/ ./scripts/
COPY setup.* ./

ENV TERM vt100
RUN guild init --yes
RUN echo "source guild-env" >> ${HOME}/.bashrc

COPY workflow/ ./workflow
COPY .git/ ./.git/
RUN bash -c 'source guild-env && pip install . && python -c "import workflow"'

ENTRYPOINT [ "bash" ]
