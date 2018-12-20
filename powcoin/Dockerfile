FROM python:3.7.0
ADD requirements.txt ./
RUN pip install -r requirements.txt
ADD utils.py ./
ADD identities.py ./
ADD my_pow_syndacoin.py ./

CMD ["python", "my_pow_syndacoin.py", "serve"]
