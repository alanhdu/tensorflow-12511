FROM tensorflow/tensorflow:1.3.0-py3

ADD . /workspace
CMD ["python", "/workspace/gen.py"]
