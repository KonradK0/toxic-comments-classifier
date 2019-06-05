FROM continuumio/miniconda3
ENTRYPOINT ["/bin/bash", "-c"]
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . .
RUN ["conda", "env", "create", "-f", "env.yml"]
ENV PATH "/opt/conda/envs/toxic-comments-classifier/bin:${PATH}"
RUN echo "source activate toxic-comments-classifier" > ~/.bashrc
RUN ["/opt/conda/envs/toxic-comments-classifier/bin/pip", "install", "-r", "requirements.txt"]
EXPOSE 8080
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"
CMD ["./start.sh"]