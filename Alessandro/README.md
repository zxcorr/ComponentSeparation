### To Install necessary packages

1) Build a environment

~$ python3 -m pip install --user virtualenv\
~$ python3 -m pip install requests

I will assume that you will create your environment in /path/to/new/virtual/environment by being /home/user/env:

~$ python3 -m venv /home/user/env\
~$ source /home/user/env/bin/activate

Now, I will assume that you requirement.txt is in /home/user/directoryCS/requirements.txt. Then,

(env) ~$ cd /home/user/directoryCS/\
(env) ~$ python3 -m pip install -r requirements.txt

It also is necessary to download gmca package from github

(env) ~$ git clone https://github.com/isab3lla/gmca4im.git\
(env) ~$ mv gmca4im/scripts/* .\
(env) ~$ rm -rf gmca4im\


### Running jupyter notebook


### Running terminal script


### Seeing results
