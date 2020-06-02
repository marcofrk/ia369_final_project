# IA 369 - 1S/2020: Final Project - Executable Papers
![Build][workflow-build] ![GitHub issues][license]

This repository contains the executable paper wrote during the IA369 - 1S/2020 lectures.
In order to reproduce it, please, follow the sections below.

## Structure

**Directories**
* [data][data]: models, images, and videos used at the executable paper
* [deliver][deliver]: the executable paper

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

---
**Note:** 
Build tested at Ubuntu 18.04.

---

### Execute the paper
Run the docker image:
```
# docker run -it --rm -p 8888:8888 ml-experience
```
Click at the returned link, as shown at the image below:
![img](data/images/docker_run.png)

[data]: https://github.com/marcofrk/ia369_final_project/tree/master/data
[deliver]: https://github.com/marcofrk/ia369_final_project/tree/master/deliver
[License]: https://img.shields.io/badge/License-GPL%20v3-blue.svg
[workflow-build]: https://github.com/diegohdorta/pyeiq/workflows/Build/badge.svg