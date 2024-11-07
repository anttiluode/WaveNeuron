import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Sensory Processor with improved webcam handling
class SensoryProcessor:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.webcam = cv2.VideoCapture(self.camera_index)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open webcam with index {self.camera_index}")
        print(f"Webcam with index {self.camera_index} opened successfully.")

    def process_frame(self):
        ret, frame = self.webcam.read()
        if not ret or frame is None:
            print("Error: Could not read frame from webcam.")
            return {'brightness': 0, 'motion': 0, 'frame': None}
        
        # Convert frame to grayscale and calculate brightness and motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        edges = cv2.Canny(gray, 100, 200)
        motion = np.mean(edges) / 255.0
        return {'brightness': brightness, 'motion': motion, 'frame': frame}

    def cleanup(self):
        """Releases the webcam resource."""
        if self.webcam:
            self.webcam.release()
            print("Webcam released.")

# Wave-based neuron class
class WaveNeuron:
    def __init__(self, base_frequency, phase_shift=0, damping=0.05):
        self.base_frequency = base_frequency
        self.phase_shift = phase_shift
        self.damping = damping

    def process_input(self, input_signal):
        # Create an output wave with frequency, phase, and damping adjustments
        return np.sin(2 * np.pi * self.base_frequency * input_signal + self.phase_shift) * np.exp(-self.damping * input_signal)

# Processing area with specialized neurons
class ProcessingArea:
    def __init__(self, num_neurons, frequency_range, area_type=""):
        self.neurons = [WaveNeuron(base_frequency=np.random.uniform(*frequency_range), phase_shift=np.random.uniform(0, np.pi)) for _ in range(num_neurons)]
        self.area_type = area_type

    def process(self, input_signal):
        # Process input signal through all neurons and combine outputs
        return sum(neuron.process_input(input_signal) for neuron in self.neurons)

# Integration of Broca and Wernicke areas in the association area
class AssociationArea:
    def __init__(self, input_areas):
        self.input_areas = input_areas

    def integrate(self, input_signal):
        integrated_output = sum(area.process(input_signal) for area in self.input_areas)
        # Synchronize outputs with an additional wave
        integration_wave = np.sin(2 * np.pi * 0.5 * input_signal)  # base frequency for synchronization
        return integrated_output * integration_wave

# Define the wave-based processing system for the bug's "brain"
class WaveBasedSystem:
    def __init__(self):
        self.eye_area = ProcessingArea(num_neurons=10, frequency_range=(0.1, 0.5), area_type="Eye")
        self.broca_area = ProcessingArea(num_neurons=10, frequency_range=(0.5, 1.0), area_type="Broca")
        self.wernicke_area = ProcessingArea(num_neurons=10, frequency_range=(1.0, 2.0), area_type="Wernicke")
        self.association_area = AssociationArea(input_areas=[self.broca_area, self.wernicke_area])

    def forward(self, input_signal):
        # Eye area transforms input
        transformed_input = self.eye_area.process(input_signal)
        # Process in Broca and Wernicke areas
        broca_output = self.broca_area.process(transformed_input)
        wernicke_output = self.wernicke_area.process(transformed_input)
        # Integrate in Association area
        final_output = self.association_area.integrate(input_signal)
        return final_output, broca_output, wernicke_output

# Main Bug system with movement logic
class BugSystem:
    def __init__(self):
        self.position = [400, 300]
        self.movement_speed = 5.0
        self.direction = 0.0  # Initial direction
        self.wave_system = WaveBasedSystem()

    def update_position(self, brightness, motion):
        # Use the wave-based brain to process brightness and motion data
        input_signal = brightness + motion
        final_output, _, _ = self.wave_system.forward(input_signal)
        
        # Calculate movement based on final output
        dx = np.cos(final_output) * self.movement_speed
        dy = np.sin(final_output) * self.movement_speed
        self.position[0] += dx
        self.position[1] += dy

        # Boundary constraints
        self.position[0] = max(50, min(750, self.position[0]))
        self.position[1] = max(50, min(550, self.position[1]))
        self.direction = np.arctan2(dy, dx)

# Main Tkinter App
class App:
    def __init__(self, root, camera_index=0):
        self.root = root
        self.root.title("Bug in the Machine with Wave-Based Brain and Cone of Vision")
        
        # Set up Bug and Sensory Processor
        self.bug = BugSystem()
        self.sensory_processor = SensoryProcessor(camera_index)
        
        # Set up Canvas
        self.canvas = tk.Canvas(root, width=800, height=600, bg="black")
        self.canvas.pack()
        
        # Start updating the bug position
        self.update_bug()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_bug(self):
        # Capture and process frame data
        features = self.sensory_processor.process_frame()
        if features['frame'] is not None:
            # Update bug position based on brightness and motion
            brightness = features['brightness']
            motion = features['motion']
            self.bug.update_position(brightness, motion)

            # Display the captured frame on the canvas
            self.draw_bug(features['frame'])

        # Schedule the next update
        self.root.after(50, self.update_bug)

    def draw_bug(self, frame):
        # Resize the frame to match the canvas dimensions
        frame_resized = cv2.resize(frame, (800, 600))
        
        # Convert the resized frame to a Tkinter-compatible image format
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Clear the canvas and draw the resized frame as background
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Keep reference to avoid GC

        # Draw the bug position as a small red circle
        x, y = self.bug.position
        self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="red")

        # Draw the cone of vision
        cone_length = 100
        cone_angle = np.pi / 6
        p1 = (x, y)
        p2 = (x + cone_length * np.cos(self.bug.direction - cone_angle), y + cone_length * np.sin(self.bug.direction - cone_angle))
        p3 = (x + cone_length * np.cos(self.bug.direction + cone_angle), y + cone_length * np.sin(self.bug.direction + cone_angle))
        
        self.canvas.create_polygon(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], fill='green', stipple='gray50')

    def on_close(self):
        # Cleanup resources on window close
        self.sensory_processor.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, camera_index=0)  # Set camera index here if not default
    root.mainloop()