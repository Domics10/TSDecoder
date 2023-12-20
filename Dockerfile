FROM --platform=arm64 ubuntu:focal

#Sistema e basilari
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libxfixes-dev \
    libx11-xcb-dev \
    libxcb-glx0-dev

#Upgrade pip
RUN pip3 install --upgrade pip && pip3 --version
COPY requirements.txt .
RUN pip3 install -r requirements.txt

#definisco l'enviroment
ENV APPDIR /home/app
WORKDIR $APPDIR

#Copio i file ed test di esecuzione
COPY . .
RUN chmod u+x ./main.py
#CMD ["python3", "./main.py"]