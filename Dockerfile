FROM python:3.10

COPY pyproject.toml poetry.lock ./

RUN pip3 install poetry
RUN poetry export -f requirements.txt -o requirements.txt --without-hashes

RUN python3.10 -m pip install -r requirements.txt
RUN python3.10 -m pip install boto3

ADD twinstat wd/twinstat

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

WORKDIR /wd
