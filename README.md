# Deep Reinforcement Learning


[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135602-b0335606-7d12-11e8-8689-dd1cf9fa11a9.gif "Trained Agents"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.  

![Trained Agents][image1]

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name deeprl python=3.6
	source activate deeprl
	```
	- __Windows__: 
	```bash
	conda create --name deeprl python=3.6 
	activate deeprl
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository, and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `deeprl` environment.  
```bash
python -m ipykernel install --user --name deeprl --display-name "deeprl"
```

5. Before running code in a notebook, change the kernel to match the `deeprl` environment by using the drop-down `Kernel` menu. 

<p align="center"><a href="https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893">
 <img width="503" height="133" src="https://user-images.githubusercontent.com/10624937/42135812-1829637e-7d16-11e8-9aa1-88056f23f51e.png"></a>
</p>
