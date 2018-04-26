# Computer Vision Nanodegree Program, Exercises

This repository contains code exercises and materials for Udacity's [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891) program. It consists of tutorial notebooks that demonstrate, or challenge you to complete, various computer vision applications and techniques. These notebooks depend on a number of software packages to run, and so, we suggest that you create a local environment with these dependencies by following the instructions below.

# Configure and Manage Your Environment with Anaconda

Per the Anaconda [docs](http://conda.pydata.org/docs):

> Conda is an open source package management system and environment management system 
for installing multiple versions of software packages and their dependencies and 
switching easily between them. It works on Linux, OS X and Windows, and was created 
for Python programs but can package and distribute any software.

## Overview
Using Anaconda consists of the following:

1. Install [`miniconda`](http://conda.pydata.org/miniconda.html) on your computer. If you already have `conda` or `miniconda` installed, you should be able to skip this step and move on to step 2.
2. Create a new `conda` [environment](http://conda.pydata.org/docs/using/envs.html) using the files in this repository.
3. Each time you wish to work on any exercises, activate your `conda` environment!

---

## 1. Installation

**Download** the latest version of `miniconda` that matches your system.

**NOTE**: There have been reports of issues creating an environment using miniconda `v4.3.13`. If it gives you issues try versions `4.3.11` or `4.2.12` from [here](https://repo.continuum.io/miniconda/).

|        | Linux | Mac | Windows | 
|--------|-------|-----|---------|
| 64-bit | [64-bit (bash installer)][lin64] | [64-bit (bash installer)][mac64] | [64-bit (exe installer)][win64]
| 32-bit | [32-bit (bash installer)][lin32] |  | [32-bit (exe installer)][win32]

[win64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
[win32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86.exe
[mac64]: https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
[lin64]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
[lin32]: https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86.sh

**Install** [miniconda](http://conda.pydata.org/miniconda.html) on your machine. Detailed instructions:

- **Linux:** http://conda.pydata.org/docs/install/quick.html#linux-miniconda-install
- **Mac:** http://conda.pydata.org/docs/install/quick.html#os-x-miniconda-install
- **Windows:** http://conda.pydata.org/docs/install/quick.html#windows-miniconda-install

## 2. Create the Environment

**Setup** the `cv-nd` environment. 

```sh
git clone https://github.com/udacity/CVND_Exercises.git
cd CVND_Exercises
```

If you are on Windows, **rename**   
`meta_windows_patch.yaml` to   
`meta.yaml`

**Create** cv-nd.  Running the command below will create a new `conda` environment that has all libraries you need to be successful in this program. This step may take a while, since you the environment is installing all the necessary packages.
```
conda env create -f environment.yaml
```

**Verify** that the cv-nd environment was created in your environments:

```sh
conda info --envs
```

**Cleanup** downloaded libraries (remove tarballs, zip files, etc):

```sh
conda clean -tp
```

### Uninstalling 

If you ever want to uninstall the environment, you can remove it by name:

```sh
conda env remove -n cv-nd
```

## 3. Use and Activate the Environment

Now that you have created an environment, you will need to activate the environment to use it! This must be done **each** time you begin a new working session i.e. open a new terminal window. 

**Activate** the `cv-nd` environment:

### OS X and Linux
```sh
$ source activate cv-nd
```
### Windows
Depending on shell either:
```sh
$ source activate cv-nd
```
or

```sh
$ activate cv-nd
```

That's it. Now all of the `cv-nd` libraries are available to you.

To exit the environment when you have completed your work session, simply close the terminal window.

