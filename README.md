# ia369_final_project - Executable Papers

This repository contains the executable paper wrote during the IA369 - 2S/2020 lectures.
In order to reproduce it, please, follow the sections below.

## Structure

**Directories**
* data: models, images, and videos used at the executable paper
* deliver: the executable paper

## How to execute the paper

### Install docker
For linux users, do the following:
```
# apt-get update
# apt-get install docker.io
```

### Build the Dockerfile
Clone this repository:
```
$ git clone https://github.com/marcofrk/ia369_final_project.git
```

Enter at the repo and build the docker container:
```
$ cd ia369_final_project
# docker build -t ml-experience .
```

[NOTE]
Build tested at Ubuntu 18.04.

### Execute the paper
Run the docker image:
```
# docker run -it --rm -p 8888:8888 ml-experience
```
Click at the returned link, as shown at the image below:
<img src="./data/images/docker_run.png"