# EVPropNet
EVPropNet: Detecting Drones By Finding Propellers For Mid-Air Landing And Following

**EVPropNet** by <a href="http://prg.cs.umd.edu"><i>Perception & Robotics Group</i></a> at the Department of Computer Science, <a href="https://umd.edu/">University of Maryland- College Park</a> and <a href="https://mavlab.tudelft.nl/"><i>Micro Air Vehicle Laboratory</i></a> at <a href="https://www.tudelft.nl/en/">Delft University of Technology</a>
.

![EVPropNet: Detecting Drones By Finding Propellers For Mid-Air Landing And Following](http://prg.cs.umd.edu/research/EVPropNet_files/Banner.jpg)
Applications presented in this work using the proposed propeller detection method for finding multi-rotors. (a) Tracking and following an unmarked quadrotor, (b) Landing/Docking on a flying quadrotor. Red and green arrows indicates the movement of the larger and smaller quadrotors respectively. Time progression is shown as quadrotor opacity. The insets show the event frames from the smaller quadrotor used for detecting the propellers of the bigger quadrotor using the proposed EVPropNet. Red and blue color in the event frames indicate positive and negative events respectively. Green color indicates the network prediction. Click on the image to zoom in.

### Abstract

The rapid rise of accessibility of unmanned aerial vehicles or drones pose a threat to general security and confidentiality. Most of the commercially available or custom-built drones are multi-rotors and are comprised of multiple propellers. Since these propellers rotate at a high-speed, they are generally the fastest moving parts of an image and cannot be directly "seen" by a classical camera without severe motion blur. We utilize a class of sensors that are particularly suitable for such scenarios called event cameras, which have a high temporal resolution, low-latency, and high dynamic range.

In this paper, we model the geometry of a propeller and use it to generate simulated events which are used to train a deep neural network called EVPropNet to detect propellers from the data of an event camera. EVPropNet directly transfers to the real world without any fine-tuning or retraining. We present two applications of our network: (a) tracking and following an unmarked drone and (b) landing on a near-hover drone. We successfully evaluate and demonstrate the proposed approach in many real-world experiments with different propeller shapes and sizes. Our network can detect propellers at a rate of 85.1% even when 60% of the propeller is occluded and can run at upto 35Hz on a 2W power budget. To our knowledge, this is the first deep learning-based solution for detecting propellers (to detect drones). Finally, our applications also show an impressive success rate of 92% and 90% for the tracking and landing tasks respectively.

- [Project Page](https://prg.cs.umd.edu/EVPropNet)
- [Paper](https://prg.cs.umd.edu/research/EVPropNet_files/EVPropNet.pdf)
- [arXiv Preprint](https://arxiv.org/abs/)

*For running the code, please read our [wiki](https://github.com/prgumd/EVPropNet/wiki).* 

## License:
Copyright (c) 2021 Perception and Robotics Group (PRG)
