# Separate legacy runtime; does not install anything into the host FK environment.
FROM python:3.8-slim-bookworm
ARG AVHUBERT_COMMIT
ARG TORCH_FLAVOR=cpu
RUN test -n "$AVHUBERT_COMMIT"
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libglib2.0-0 libgl1 libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pip==23.3.2 setuptools==59.5.0 wheel==0.41.3 Cython==0.29.36 numpy==1.23.5
RUN pip install --no-cache-dir torch==1.13.1+${TORCH_FLAVOR} torchvision==0.14.1+${TORCH_FLAVOR} torchaudio==0.13.1+${TORCH_FLAVOR} --extra-index-url https://download.pytorch.org/whl/${TORCH_FLAVOR}
RUN git clone --no-checkout https://github.com/facebookresearch/av_hubert.git /opt/av_hubert && cd /opt/av_hubert && git checkout --detach "$AVHUBERT_COMMIT" && git submodule update --init --recursive
RUN pip install --no-cache-dir hydra-core==1.0.7 omegaconf==2.0.6 PyYAML==6.0.1 cffi regex sacrebleu==2.4.3 tqdm bitarray==2.9.2 && pip install --no-cache-dir --no-build-isolation --no-deps -e /opt/av_hubert/fairseq
RUN pip install --no-cache-dir python-speech-features==0.6 scipy==1.10.0 opencv-python-headless==4.5.5.64 sentencepiece==0.1.99 editdistance==0.6.2 av==12.3.0 Pillow==10.2.0 facenet-pytorch==2.5.3 scikit-learn==1.3.2 soundfile==0.13.1
ENV PYTHONPATH=/workspace:/workspace/src
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "baselines.avtenet"]
