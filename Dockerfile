FROM harbor.weizhipin.com/arsenal-ai/easydel:nightly
RUN apt update && apt install -y openmpi-bin openmpi-common libopenmpi-dev
