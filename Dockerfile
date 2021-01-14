FROM ubuntu:18.04
COPY . .
RUN scripts/lambda.sh