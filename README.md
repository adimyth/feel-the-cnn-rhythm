# Notes

* https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp39-cp39-linux_x86_64.whl
* https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl

```
If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false
```

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
  