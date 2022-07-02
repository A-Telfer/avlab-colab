#!/bin/bash

VERSION=1.0

usage() { echo "Usage: $0 [-p (push)]" 1>&2; exit 1; }

push=0
while getopts ":p" o; do
    echo ${OPTARG}
    case "${o}" in
        p)
            push=1
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

docker build . \
    --tag "andretelfer/avlab-lsd-oft-analysis:$VERSION" \
    --tag "andretelfer/avlab-lsd-oft-analysis:latest"

if [ ${push} == 1 ]; then
    docker push "andretelfer/avlab-lsd-oft-analysis:$VERSION"
    docker push "andretelfer/avlab-lsd-oft-analysis:latest"
fi


