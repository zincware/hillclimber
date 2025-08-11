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

# TODO: you can do great committer analysis, finding all the structures close to the TS CVs and then use laufband to run short MD with all those and all of them with different velocity initializations and check if they go to educt / product. The ratio / the committor should be close to 1:1 for the TS CVs.
