FROM continuumio/miniconda3

ADD . /usr/local

WORKDIR /usr/local

RUN python setup.py bdist_wheel && \
    pip install dist/event_predictor-0.1-py3-none-any.whl
	
WORKDIR /usr/local/build/lib/project