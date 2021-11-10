#!/bin/bash

image_name="kws_template"
username="ISedunov"
container_name=${username}-${image_name}

docker stop "${container_name}"
docker rm "${container_name}"

docker run -it \
    --gpus all \
    --expose 22 -P \
    --shm-size 8G \
    --runtime=nvidia \
    -v $PWD/../../:/home/key_word_spotting \
    -v /mount/export0:/mount/export0 \
    -v /mount/export2:/mount/export2 \
    -v /mount/export1:/mount/export1 \
    -v /mount/export3:/mount/export3 \
    --detach \
    --name "${container_name}" \
    --entrypoint /bin/bash \
    ${image_name}