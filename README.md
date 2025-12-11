# pyWFC3 package

This is a place holder for all the codes and scripts that help in reducing
and anylysing HST's WFC3 data.

## Installation.
1. Create an conda environment to install this package.
```shell
conda create --name pywfc3_utils python=3.11
```

2. Activate the conda environment
```shell
conda activate pywfc3_utils
```

<!-- 
3. Clone the pyWFC3 package.
```setup
git clone git@github.com:pyAstroDude/pyWFC3.git
```
 -->

3. Install package using pip. I have used the HTTP protocol to get the package but 
you can also use SSH if you prefer.
```shell
pip install git+https://github.com/pyAstroDude/pyWFC3.git
```

4. The select_data gui uses SAOImageDS9 to display the data. The GUI communicates 
with DS9 using the XPA messaging system. So you need to set up the correct XPA
method. You can configure this in your shell's config file. For example, if you
are using a bash shell then edit your .basrc or .bash_profile file to include the 
following:
```shell
export XPA_METHOD=local
```
Don't forget to source you config file before using the selec_data gui.
