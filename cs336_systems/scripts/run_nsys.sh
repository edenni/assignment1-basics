uv run /usr/local/bin/nsys profile \
    --trace=cuda,cublas,cudnn,nvtx,osrt \
    --python-backtrace=cuda \
    --pytorch=functions-trace,autograd-nvtx \
    --force-overwrite=true \
    -o result \
    python cs336_systems/benchmark.py