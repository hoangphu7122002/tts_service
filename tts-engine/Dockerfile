FROM gpuci/miniconda-cuda:10.2-base-ubuntu18.04

WORKDIR /app

# Install build packages
RUN apt-get update \
  && apt-get install -y build-essential python-dev

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml


# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "tts", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure torch is installed:"
RUN python -c "import torch; print(torch.cuda.is_available())"

# Install zag2p using Deploy token
RUN pip install git+https://ai:X7cS2sUk4Non6_D-1syH@gitlab.com/aic-tts/zag2p.git@dev

# Install JRE 8 for VnCoreNLP
RUN apt-get install -y openjdk-8-jre

RUN java -version

# Add CXX Lib for pytorch
RUN echo "export LD_LIBRARY_PATH=/opt/conda/envs/tts/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

SHELL ["/bin/bash", "-c"]

RUN echo "conda activate tts" >> ~/.bashrc

# Remove build packages
RUN apt-get remove -y build-essential python-dev \
  && apt-get autoremove -y
