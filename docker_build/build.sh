#!/bin/bash

docker build . \
  --build-arg http_proxy="${http_proxy}" \
  --build-arg https_proxy="${https_proxy}" \
  --build-arg all_proxy="${all_proxy}" \
  --build-arg no_proxy="${no_proxy}" \
  -t hub.i.basemind.com/test/f5-tts-grpo:build_test \
  -f dockerfile