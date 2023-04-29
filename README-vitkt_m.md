### VOT2022-LT submission - VITKT_M
Jie Zhao, Xin Chen, Chang Liu, Houwen Peng, Dong Wang, Huchuan Lu

Dalian University of Technology, China  

##### Hardware and OS specifications
CPU: Intel Core i9-9900KF CPU @3.60GHz × 16 
Memory: 64GB
GPU: NVIDIA GeForce RTX 2080 Ti
GPU Memory: 11GB
OS: Ubuntu 18.04  

#### VOT2022-LT test instructions
To run the VOT2022-LT experiments please follow these instructions:

+ Download the pre-trained model files ``checkpoints.zip`` from [here](https://drive.google.com/file/d/14IUGPwhKlLjiNoFMAAiJG9qMxup9dB7S/view?usp=sharing) and extract it in the folder ``vitkt_m/``

+ Move to the submission source folder ``cd vitkt_m``

+ Create the Anaconda environment ``conda create -n vitkt_m python=3.6``

+ Activate the environment ``conda activate vitkt_m``

+ Install the environment ``pip install -r requirements.txt``

+ Install ninja-build ``sudo apt-get install ninja-build``

+ Edit the variable ``base_path`` in the file ``vot_path.py`` by providing the full-path to the location where the submission folder is stored,
and edit the paths ``[full-path-to-vitkt_m]`` and ``[env-path-to-vitkt_m]`` in the file ``trackers.ini`` (in line 6 and 7).
 
+ Run ``python test_env.py`` to verify whether the environment is configured well.
 
+ Run the evaluation by ``vot evaluate VITKT_M`` 

+ Run the analysis by ``vot analysis VITKT_M``  

#### If you fail to run our tracker please contact ``zj982853200@mail.dlut.edu.cn``
