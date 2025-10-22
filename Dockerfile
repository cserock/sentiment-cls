#Python 3.11.9
# Base image
FROM python:3.11.9

RUN apt-get update
RUN apt-get -y install build-essential libc-dev gcc python3-dev git

# Set working directory
WORKDIR /app

# Install python lib
RUN pip install --upgrade pip
RUN pip install python-dotenv
RUN pip install --no-cache-dir torch==2.0.1
RUN pip install kiwipiepy==0.19.0

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt --timeout 10000

# Copy files
COPY ./main.py /app/
COPY ./packages /app/packages

CMD ["python3", "main.py", "--host", "0.0.0.0", "--port", "8088"]