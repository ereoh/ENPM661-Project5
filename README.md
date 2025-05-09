# Project 5: Implementing ARRT-Connect

Grayson Gilbert  
UID: 115837461  
Directory ID: ggilbert  

Marcus Hurt  
UID: 121361738  
Directory ID: mhurt  

Erebus Oh  
UID: 117238105  
Directory: eoh12  

ENPM 661, Spring 2025  

Phase 1: TBD  
Phase 2: May 13, 2025  

[Github Link](https://github.com/ereoh/ENPM661-Project5)  

Overleaf Link: https://www.overleaf.com/5468471675mqthqhcxyqgp#4e737a  

[OMPL Github Link](https://github.com/robotic-esp/ompl)  

[An Adaptive Rapidly-Exploring Random Tree Paper](https://ieeexplore.ieee.org/document/9536671)  

## Run This Code
---

### Dependencies
Dependencies used in this project include: numpy, pygame, heapq, math, and time.

Install the necessary dependencies:
```bash
pip install numpy matplotlib 
```

### Launch Gazebo Simulation
To launch the Gazebo simulation with the TurtleBot3 model, follow these steps:

1. Build the workspace:
   ```bash
   colcon build
   ```

2. Source the setup file:
   ```bash
   source install/setup.bash
   ```

3. Set the TurtleBot3 model environment variable:
   ```bash
   export TURTLEBOT3_MODEL=waffle
   ```

4. Launch the Gazebo world:
   ```bash
   ros2 launch gazebo_worlds turtlebot3_world.launch.py
   ```

Once launched, the simulation will start with the TurtleBot3 model in the Gazebo environment.

To change the spawn position, simply update the spawn_config.txt file in the format:
```
x y theta
```

### Run the ROS 2 Node
To run the `arrt_connect_node` and find a path using the ARRT-Connect algorithm, execute the following command:

```bash
ros2 run arrt_connect arrt_connect_node
```

To update the start and goal positions, change them in the arrt_connect_node.py

## Experiments
The experiments for this project were run without the simulation as the interest was in the planning aspect of ARRT-Connect.

This section explains how to reproduce the experiments.

# Expriment 1
1. Set r on line 12 of search.py to 0.

2. Set map_instance on line 146 of search.py to Map(BUFFER=0).

3. Set qinit on line 715 of search.py to (40, 50)

4. Set qinit on line 716 of search.py to (40, 250)

5. Run search.py
```
python3 search.py
```

# Expriment 2
1. Set r on line 12 of search.py to 22.

2. Set map_instance on line 146 of search.py to Map(BUFFER=10).

3. Set qinit on line 715 of search.py to (50, 50)

4. Set qinit on line 716 of search.py to (500, 50)

5. Run search.py
```
python3 search.py
```


# Expriment 3
1. Set r on line 12 of search.py to 22.

2. Set map_instance on line 146 of search.py to Map(BUFFER=20).

3. Set qinit on line 715 of search.py to (50, 50)

4. Set qinit on line 716 of search.py to (500, 50)

5. Run search.py
```
python3 search.py
```

# Expriment 4
1. Set r on line 12 of search.py to 22.

2. Set map_instance on line 146 of search.py to Map(BUFFER=20).

3. Set qinit on line 715 of search.py to (50, 50)

4. Set qinit on line 716 of search.py to (500, 50)

5. Comment out lines 73-83 of search.py (the first and second walls).

6. Comment out lines 97-101 of search.py (the fourth wall).

5. Run search.py
```
python3 search.py
```
