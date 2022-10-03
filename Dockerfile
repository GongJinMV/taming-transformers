FROM pytorch/pytorch:latest

LABEL maintainer="jin.gong"

RUN apt-get update && apt-get install libglib2.0-dev