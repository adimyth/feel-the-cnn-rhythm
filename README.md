# Notes

* https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp39-cp39-linux_x86_64.whl
* https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl

```
If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false
```

# Installing torch and related packages (don't do this)

* Install [`light-the-torch`](https://github.com/pmeier/light-the-torch/issues/32) WITH pip: `poetry run pip install light-the-torch`
* `poetry run ltt install torch`
* `poetry run pip install torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
* For future follow-up
  * https://stackoverflow.com/questions/64871630/how-to-convert-pip-install-to-poetry-file
  * https://github.com/python-poetry/poetry/issues/4124
  * https://github.com/pmeier/light-the-torch
  * https://stackoverflow.com/questions/59158044/poetry-and-pytorch
  * https://github.com/python-poetry/poetry/issues/1616

/home/soumendra/project-zoo/feel-the-cnn-rhythm/.venv/lib/python3.8/site-packages/pytorch_lightning/core/datamodule.py:423: LightningDeprecationWarning: DataModule.setup has already been called, so it will not be called again. In v1.6 this behavior will change to always call DataModule.setup.

pytorch_lightning.utilities.exceptions.MisconfigurationException: `.test(ckpt_path="best")` is set but `ModelCheckpoint` is not configured to save the best model.

# Installing docker-compose

* https://docs.docker.com/compose/install/

```bash
# sudo apt install py-pip python3-dev libffi-dev openssl-dev gcc libc-dev rust cargo make
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
docker-compose --version
```


# Libraries

* [pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
* [lightning-flash](https://lightning-flash.readthedocs.io/en/latest/quickstart.html)
* [dlint](https://github.com/duo-labs/dlint)
* [pytorch binaries](https://download.pytorch.org/whl/torch_stable.html)
  