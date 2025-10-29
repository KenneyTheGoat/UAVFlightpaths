# UAV Flight Trajectory Visualization System

## Description
This application provides a comprehensive 3D visualization environment for Unmanned Aerial Vehicle (UAV) flight trajectories. It features 3D model loading and rendering using Open3D, trajectory planning and waypoint management, real-time flight simulation with FPV (First Person View), video recording capabilities, interactive camera controls, and trajectory analysis. Built with Tkinter for the GUI, Open3D for rendering, and supporting libraries like NumPy, Trimesh, SciPy, OpenCV, and Pillow.

## Requirements
- Python 3.9+
- Dependencies: `pip install numpy open3d trimesh scipy opencv-python pillow`
- Operating System: Windows 10/11 or Linux Ubuntu 18.04+
- Memory: 8GB recommended
- Graphics: Dedicated GPU with OpenGL 3.3+ support, but can run on cpu

## Installation
1. Clone the repository:
   
   git clone https://gitlab.cs.uct.ac.za/capstone-20255/visualizing-and-measuring-optimal-uav-flight-paths.git
   cd visualizing-and-measuring-optimal-uav-flight-paths
   
2. Install dependencies:
  
   pip install -r requirements.txt
   `

## Usage
### `vis.py`
Main GUI application for UAV trajectory visualization:

python vis.py

- Launches a Tkinter-based interface with collapsible panels for model loading, trajectory management, waypoint editing, camera controls, flight simulation, and recording.
- Interactive 3D view powered by Open3D, waypoint list, FPV preview, and trajectory metrics display.

## User Manual
User manual is provided together with documentation on seperate document

## Collaborate with your team
1. Max Mkhabela 
2. Kenneth Baloyi 
3. Lethabo Neo 

## Support
For issues or questions, open an issue on the GitLab project.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Project Status
Actively developed as part of a capstone project. This version integrates Tkinter with Open3D for improved rendering and interactivity.
