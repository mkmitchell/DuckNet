FROM ubuntu:latest
ENV TERM xterm

RUN apt-get update \
 && apt-get install -y curl wget nano git unzip psmisc \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh -O miniconda.sh \
 && bash miniconda.sh -b -p /root/miniconda   \
 && rm   miniconda.sh
ENV PATH="/root/miniconda/bin:${PATH}"

RUN apt-get -y install xvfb x11vnc openbox psmisc             \
 && apt-get -y install libgtk-3.0 libasound2 libnss3 libgbm1

WORKDIR /root
RUN wget https://github.com/clickot/ungoogled-chromium-binaries/releases/download/99.0.4844.51-1.1/ungoogled-chromium_99.0.4844.51-1.1.AppImage -O chromium.AppImage \
 && wget https://gitlab.com/api/v4/projects/24386000/packages/generic/librewolf/97.0.1-1/LibreWolf.x86_64.AppImage -O librewolf.AppImage \
 && chmod +x *.AppImage \
 && ./chromium.AppImage  --appimage-extract && mv squashfs-root chromium   \
 && ./librewolf.AppImage --appimage-extract && mv squashfs-root librewolf  \
 && rm ./*.AppImage


RUN pip install seleniumbase && sbase install chromedriver 98 --path && sbase install geckodriver --path  \
 && pip install pyppeteer coverage
ADD requirements.txt ./
RUN pip install -r requirements.txt && rm requirements.txt

ADD        tests/docker/chromium /usr/bin/
ADD        tests/docker/firefox  /usr/bin/
ADD        tests/docker/start_x11.sh ./
ENTRYPOINT ["./start_x11.sh"]

#make new files created by root user r+w for others
RUN echo "umask 0000\n" >> /root/.bashrc

ENV DIG_IT_TEST_DOCKER=1

CMD /bin/bash
