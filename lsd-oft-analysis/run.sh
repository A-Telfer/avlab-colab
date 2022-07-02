export AVLAB_OFT_DATA=/home/andretelfer/shared/curated/vern/

docker run \
    -it --rm \
    --shm-size 4GB \
    --network host \
    -v `pwd`:/home/jovyan/shared \
    -v $AVLAB_OFT_DATA:/home/jovyan/data \
    andretelfer/avlab-lsd-oft-analysis:latest \
    bash 