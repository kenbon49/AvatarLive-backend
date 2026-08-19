# Local Runtime Assets

Store every runtime asset that is not part of a CosyVoice checkpoint in this
directory. The Compose service sets `MODELSCOPE_CACHE` to `local_models/modelscope`
and enables `MODELSCOPE_OFFLINE=1`, so runtime model downloads are prohibited.

The Wetext resource must be present at:

`local_models/modelscope/hub/pengzhendong/wetext`

To preserve an already downloaded copy from the existing container, run:

```bash
mkdir -p CosyVoice/local_models/modelscope/hub/pengzhendong
docker cp cosyvoice_dev:/root/.cache/modelscope/hub/pengzhendong/wetext \
  CosyVoice/local_models/modelscope/hub/pengzhendong/
```

If the directory is absent, CosyVoice must not be started until it has been
copied or downloaded explicitly into this project directory.
