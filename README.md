For `uv add plumed` use

```bash
export CC=gcc
export CXX=g++
```

and use something like the following `env.yaml`.

```yaml
global:
  PLUMED_KERNEL: /data/fzills/tools/plumed2/bin/lib/libplumedKernel.so
```