## Bug in the Machine with Wave-Based Brain and Cone of Vision

Video about my crazy wave neuron filled day today: **https://www.youtube.com/watch?v=jymidvWRBWM**

Wont be up for long probably. 

This project features a simulated "bug" driven by a wave-based neuron model inspired by biological systems. The system uses a webcam as a sensory input to detect motion and brightness in the environment and makes movement decisions accordingly. This README provides an overview of how to set up, run, and understand the core components of the project.

## Overview

The project simulates a bug with a "brain" consisting of wave-based processing areas for decision-making and movement control. It uses a webcam to capture visual information and processes this information using specialized neurons modeled with sinusoidal responses.

## Key Features:

Webcam Integration: The system captures frames from a connected webcam and processes brightness and motion.

Wave-Based Neurons: The "brain" of the bug is based on WaveNeuron classes which simulate signal processing similar to how biological neurons function.

Tkinter GUI: The simulation is visualized in a Tkinter window, where a red dot represents the bug, and a "cone of vision" indicates its direction of focus.

## Requirements

The project requires Python 3.x and the following dependencies:

opencv-python (cv2) for capturing webcam input

numpy for numerical calculations

Pillow for image handling within Tkinter

tkinter for GUI visualization

To install the dependencies, use the following command:

pip install opencv-python numpy Pillow

## How to Run the Project

1. Clone the Repository

First, clone this repository or download the project files to your local machine.

2. Connect a Webcam

Ensure that a webcam is connected to your system. The default camera_index is set to 0, but you can change it if you have multiple webcams.

3. Run the Script

Execute the Python script to launch the Tkinter GUI and start the simulation.

app.py 

4. Close the App

To exit the application, close the Tkinter window. This will also release the webcam resource.

## Detailed Explanation of Components

## SensoryProcessor Class

This class is responsible for interfacing with the webcam and extracting sensory information from each frame. The extracted features include:

Brightness: Calculated as the average intensity of the grayscale version of the frame.

Motion: Extracted using edge detection (cv2.Canny) to approximate areas with significant changes.

## WaveNeuron Class

The wave-based neurons are modeled using sinusoidal functions that simulate natural oscillations in response to stimuli. Each neuron has a:

Base frequency: Determines the speed of oscillation.

Phase shift: Adds variability to neuron responses.

Damping factor: Causes the response to decay over time.

## Processing Areas and Association Area

Eye Area: Processes sensory input data related to brightness and motion.

Broca and Wernicke Areas: These areas simulate different types of higher processing, akin to speech and language processing in the human brain.

Association Area: Integrates input from Broca and Wernicke areas for complex processing.

## Bug System

The BugSystem class controls the bug's movement based on the sensory input processed by the wave-based neurons. The bug moves in a simulated environment represented in the Tkinter GUI.

Movement Logic: The bug's position is updated based on the processed output from the wave neurons, taking into account both brightness and motion from the sensory processor.

## Tkinter GUI Visualization

Canvas: The bug's movement is displayed on an 800x600 canvas.

Bug Representation: The bug is shown as a red circle that moves across the canvas based on the outputs from the wave-based brain.

Cone of Vision: A green cone represents the direction the bug is "looking" based on its current motion.

## Customization

Webcam Index: If the default webcam index (0) does not work, you can modify camera_index when initializing the SensoryProcessor or in the App class constructor.

Neurons and Frequency Ranges: You can experiment with the number of neurons or their frequency ranges in the ProcessingArea class to observe different movement behaviors.

## Troubleshooting

Webcam Not Found: If the script cannot find the webcam, ensure the webcam is properly connected and accessible. You may need to change the camera_index in the script.

Performance Issues: The script captures frames at a regular interval. If performance is slow, try reducing the canvas size or increasing the interval in the update_bug() function by modifying the after() value.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute the code.

## Acknowledgments

Special thanks to the open-source community for providing tools like OpenCV, Numpy, and Tkinter, which made this project possible.

## Future Improvements

Enhanced AI: The current bug movement is simple and deterministic. Future iterations could integrate more complex decision-making models using machine learning.

Collision Detection: Implement collision detection to create a more interactive environment for the bug to navigate.

Multi-Bug Environment: Introduce multiple bugs to simulate interactions and group behaviors.

