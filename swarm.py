import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
from dataclasses import dataclass
from collections import deque
import random
import logging
from typing import List, Tuple, Optional

@dataclass
class WaveFlyConfig:
    # Vision settings
    vision_cone_angle: float = np.pi / 3
    vision_cone_length: int = 150
    vision_width: int = 32
    vision_height: int = 32

    # Camera settings
    camera_width: int = 1280
    camera_height: int = 720

    # Movement settings
    max_speed: float = 8.0
    turn_speed: float = 0.15
    hover_amplitude: float = 2.0
    
    # Population settings
    initial_population: int = 12
    max_population: int = 30

    # Visual settings
    fly_size: float = 5.0
    fly_visibility: int = 5

class WaveNeuron:
    def __init__(self, base_frequency, phase_shift=0, damping=0.05):
        self.base_frequency = base_frequency
        self.phase_shift = phase_shift
        self.damping = damping
        self.time = 0

    def process_input(self, input_signal, dt=0.1):
        self.time += dt
        return (np.sin(2 * np.pi * self.base_frequency * input_signal + self.phase_shift) * 
                np.exp(-self.damping * self.time))

class ProcessingArea:
    def __init__(self, num_neurons, frequency_range, area_type=""):
        self.neurons = [
            WaveNeuron(
                base_frequency=np.random.uniform(*frequency_range),
                phase_shift=np.random.uniform(0, np.pi)
            ) for _ in range(num_neurons)
        ]
        self.area_type = area_type
        self.output_history = deque(maxlen=30)

    def process(self, input_signal):
        outputs = [n.process_input(input_signal) for n in self.neurons]
        combined = np.mean(outputs)
        self.output_history.append(combined)
        return combined

class AssociationArea:
    def __init__(self, input_areas):
        self.input_areas = input_areas
        self.integration_frequency = 0.5
        self.phase = 0
        self.memory = deque(maxlen=30)

    def integrate(self, input_signal):
        integrated_output = sum(area.process(input_signal) for area in self.input_areas)
        integration_wave = np.sin(2 * np.pi * self.integration_frequency * input_signal + self.phase)
        result = integrated_output * integration_wave
        self.memory.append(result)
        return result

class EnhancedWaveBrain:
    def __init__(self):
        # Sensory processing
        self.eye_area = ProcessingArea(num_neurons=12, frequency_range=(0.1, 0.5), area_type="Eye")
        
        # Higher-order processing
        self.broca_area = ProcessingArea(num_neurons=10, frequency_range=(0.5, 1.0), area_type="Broca")
        self.wernicke_area = ProcessingArea(num_neurons=10, frequency_range=(1.0, 2.0), area_type="Wernicke")
        
        # Integration
        self.association_area = AssociationArea([self.broca_area, self.wernicke_area])
        
        # State memory
        self.memory = deque(maxlen=30)
        self.wave_state = {
            'visual': deque(maxlen=10),
            'motion': deque(maxlen=10),
            'decision': deque(maxlen=10)
        }

    def process_vision(self, vision_input: np.ndarray) -> Tuple[float, float]:
        input_intensity = np.mean(vision_input) / 255.0
        
        visual_signal = self.eye_area.process(input_intensity)
        self.wave_state['visual'].append(visual_signal)
        
        broca_output = self.broca_area.process(visual_signal)
        wernicke_output = self.wernicke_area.process(visual_signal)
        self.wave_state['motion'].append(broca_output)
        
        decision = self.association_area.integrate(visual_signal)
        self.wave_state['decision'].append(decision)
        
        velocity = np.tanh(broca_output + decision)
        rotation = np.tanh(wernicke_output - visual_signal)
        
        return velocity, rotation

class FlyVisuals:
    def __init__(self):
        self.mating_glow_phase = 0.0
        
    def is_within_frame(self, x, y, frame) -> bool:
        height, width = frame.shape[:2]
        return 0 <= x < width and 0 <= y < height
    
    def draw_fly(self, frame, x, y, angle, size=5.0, shade=0.7, wave_state=0.0):
        x, y = int(x), int(y)
        
        # Wing animation with wave influence
        wing_phase = (time.time() % (2 * np.pi)) * 3
        wing_amplitude = 0.5 + 0.2 * abs(wave_state)
        
        # Body parameters modulated by wave state
        body_length = size * (1.2 + 0.1 * wave_state)
        body_width = max(2, int(size / 1.8))
        
        # Draw segmented body
        segments = 3
        for i in range(segments):
            segment_ratio = i / segments
            segment_width = max(1, int(body_width * (1.0 - segment_ratio * 0.3)))
            segment_x = x - body_length * np.cos(angle) * segment_ratio
            segment_y = y - body_length * np.sin(angle) * segment_ratio
            
            # Color influenced by wave state
            base_shade = int(shade * 255)
            color = (
                int(base_shade * (0.9 + 0.1 * wave_state)),
                int(base_shade * (1.0 + 0.1 * wave_state)),
                int(base_shade * (1.1 + 0.1 * wave_state))
            )
            
            if self.is_within_frame(segment_x, segment_y, frame):
                cv2.circle(frame, 
                          (int(segment_x), int(segment_y)), 
                          segment_width,
                          color, 
                          -1)

        # Draw wings with wave-influenced movement
        for wing_side in [-1, 1]:
            wing_angle = angle + np.pi/2 * wing_side + np.sin(wing_phase) * wing_amplitude
            wing_length = size * (1.5 + 0.2 * abs(wave_state))
            
            wing_points = np.array([
                [x, y],
                [int(x + wing_length * 0.7 * np.cos(wing_angle - 0.2)),
                 int(y + wing_length * 0.7 * np.sin(wing_angle - 0.2))],
                [int(x + wing_length * np.cos(wing_angle)),
                 int(y + wing_length * np.sin(wing_angle))],
                [int(x + wing_length * 0.7 * np.cos(wing_angle + 0.2)),
                 int(y + wing_length * 0.7 * np.sin(wing_angle + 0.2))]
            ], dtype=np.int32)
            
            cv2.fillPoly(frame, [wing_points], (240, 240, 240))
            
            vein_end_x = int(x + wing_length * np.cos(wing_angle))
            vein_end_y = int(y + wing_length * np.sin(wing_angle))
            if self.is_within_frame(vein_end_x, vein_end_y, frame):
                cv2.line(frame, (x, y), 
                        (vein_end_x, vein_end_y),
                        (180, 180, 180), 1)

        # Draw wave-influenced antennae
        antenna_segments = 3
        for ant_side in [-1, 1]:
            prev_x, prev_y = x, y
            ant_angle = angle + np.pi/6 * ant_side
            segment_length = size * 0.3
            
            for seg in range(antenna_segments):
                ant_angle += np.sin(time.time() * 2 + seg) * (0.1 + 0.05 * wave_state)
                end_x = int(prev_x + segment_length * np.cos(ant_angle))
                end_y = int(prev_y + segment_length * np.sin(ant_angle))
                if self.is_within_frame(end_x, end_y, frame):
                    cv2.line(frame, (prev_x, prev_y), (end_x, end_y), color, 1)
                    prev_x, prev_y = end_x, end_y

        # Draw vision cone with wave modulation
        vision_length = 100 * (1 + 0.2 * wave_state)
        vision_angle = np.pi / 3 * (1 + 0.1 * wave_state)
        left_angle = angle - vision_angle / 2
        right_angle = angle + vision_angle / 2
        
        wave_phase = time.time() * 2
        vision_points = []
        num_segments = 10
        
        for i in range(num_segments + 1):
            t = i / num_segments
            current_angle = left_angle * (1 - t) + right_angle * t
            r = vision_length * (1 + 0.1 * np.sin(wave_phase + t * 4 * np.pi))
            px = x + r * np.cos(current_angle)
            py = y + r * np.sin(current_angle)
            vision_points.append([int(px), int(py)])
        
        vision_points = np.array([[[x, y]] + vision_points], dtype=np.int32)
        cv2.polylines(frame, vision_points, True, 
                     (int(100 * (1 + wave_state)), 
                      255, 
                      int(255 * (1 + wave_state)), 50), 1)

class WaveFly:
    def __init__(self, x: float, y: float, config: WaveFlyConfig):
        self.x = x
        self.y = y
        self.angle = random.uniform(0, 2 * np.pi)
        self.config = config
        self.brain = EnhancedWaveBrain()
        
        # Motion state
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.hover_offset = random.uniform(0, 2 * np.pi)
        
        # Neural oscillation influence
        self.wave_phase = random.uniform(0, 2 * np.pi)
        self.broca_influence = 0.0
        self.wernicke_influence = 0.0
        
        # Screen boundaries
        self.screen_width = 0
        self.screen_height = 0
    
    def get_vision_cone(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Calculate vision cone points
        angle = self.config.vision_cone_angle
        length = self.config.vision_cone_length
        left_angle = self.angle - angle / 2
        right_angle = self.angle + angle / 2
        
        points = np.array([
            [int(self.x), int(self.y)],
            [int(self.x + length * np.cos(left_angle)), 
             int(self.y + length * np.sin(left_angle))],
            [int(self.x + length * np.cos(right_angle)), 
             int(self.y + length * np.sin(right_angle))]
        ], dtype=np.int32)
        
        cv2.fillPoly(mask, [points], 255)
        cone_image = cv2.bitwise_and(frame, frame, mask=mask)
        cone_resized = cv2.resize(cone_image, 
                                (self.config.vision_width, 
                                 self.config.vision_height))
        
        return cone_resized
        
    def update(self, frame: np.ndarray):
        # Get visual input
        vision = self.get_vision_cone(frame)
        
        # Get brain's decisions
        velocity, rotation = self.brain.process_vision(vision)
        
        # Get wave influences
        broca_state = np.mean([x for x in self.brain.broca_area.output_history]) if self.brain.broca_area.output_history else 0
        wernicke_state = np.mean([x for x in self.brain.wernicke_area.output_history]) if self.brain.wernicke_area.output_history else 0
        
        # Update movement with neural oscillations
        self.wave_phase += 0.1
        wave_factor = np.sin(self.wave_phase)
        
        # Apply brain area influences
        self.velocity = (velocity * self.config.max_speed * 
                        (1 + 0.2 * wave_factor) * 
                        (1 + 0.3 * broca_state))
        
        self.angular_velocity = (rotation * self.config.turn_speed * 
                               (1 + 0.1 * wave_factor) * 
                               (1 + 0.3 * wernicke_state))
        
        # Add hovering motion modulated by brain state
        hover_strength = 1 + 0.5 * abs(broca_state)
        hover_y = np.sin(time.time() + self.hover_offset) * self.config.hover_amplitude * hover_strength
        
        # Update position with all influences
        self.x += self.velocity * np.cos(self.angle)
        self.y += self.velocity * np.sin(self.angle) + hover_y
        self.angle += self.angular_velocity
        
        # Screen wrapping
        if self.screen_width and self.screen_height:
            self.x = self.x % self.screen_width
            self.y = self.y % self.screen_height
        
        return np.mean([broca_state, wernicke_state])  # Return average wave state for visualization

class WaveFlySimulation:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wave Brain Flies")
        
        # Initialize config and flies list
        self.config = WaveFlyConfig()
        self.flies = []
        
        # Setup camera
        self.setup_camera()
        
        # Create GUI elements
        self.setup_gui()
        
        # Initialize population
        self.initialize_population()
        
        # Create fly visuals handler
        self.fly_visuals = FlyVisuals()
        
        # Processing state
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.paused = False
        
        # Start processing
        self.start_processing()
    
    def setup_camera(self):
        """Initialize the webcam"""
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            if not self.camera.isOpened():
                raise RuntimeError("Could not open camera")
                
            # Get actual dimensions
            self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            raise
    
    def setup_gui(self):
        """Setup the GUI elements"""
        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Population control
        ttk.Label(control_frame, text="Population:").pack(side=tk.LEFT, padx=5)
        self.population_var = tk.StringVar(value="0")
        ttk.Label(control_frame, textvariable=self.population_var).pack(side=tk.LEFT, padx=5)
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_scale = ttk.Scale(
            control_frame,
            from_=1.0,
            to=15.0,
            orient=tk.HORIZONTAL,
            value=self.config.max_speed,
            command=self.update_speed
        )
        self.speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Wave influence control
        ttk.Label(control_frame, text="Wave Influence:").pack(side=tk.LEFT, padx=5)
        self.wave_scale = ttk.Scale(
            control_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            value=0.5,
            command=self.update_wave_influence
        )
        self.wave_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Pause button
        self.pause_button = ttk.Button(
            control_frame, 
            text="Pause", 
            command=self.toggle_pause
        )
        self.pause_button.pack(side=tk.RIGHT, padx=5)
        
        # Canvas for video display
        self.canvas = tk.Canvas(
            self.root, 
            width=self.width, 
            height=self.height,
            bg='black'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def initialize_population(self):
        """Create initial fly population"""
        self.flies = []
        margin = 50
        for _ in range(self.config.initial_population):
            fly = WaveFly(
                x=random.uniform(margin, self.width - margin),
                y=random.uniform(margin, self.height - margin),
                config=self.config
            )
            fly.screen_width = self.width
            fly.screen_height = self.height
            self.flies.append(fly)
        
    def update_speed(self, val):
        """Update fly speed from slider"""
        self.config.max_speed = float(val)
        
    def update_wave_influence(self, val):
        """Update wave influence strength"""
        for fly in self.flies:
            for area in [fly.brain.broca_area, fly.brain.wernicke_area]:
                for neuron in area.neurons:
                    neuron.damping = 0.05 * (1 - float(val))
    
    def toggle_pause(self):
        """Toggle simulation pause state"""
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")
    
    def start_processing(self):
        """Start the processing threads"""
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        self.process_frame()
    
    def camera_loop(self):
        """Background thread for camera capture"""
        while self.running:
            if not self.paused:
                ret, frame = self.camera.read()
                if ret and not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(1/30)
    
    def process_frame(self):
        """Main processing loop"""
        if not self.paused and not self.frame_queue.empty():
            frame = self.frame_queue.get()
            
            # Update all flies
            for fly in self.flies:
                wave_state = fly.update(frame)
                
            # Update display
            self.update_display(frame)
            
            # Update stats
            self.population_var.set(str(len(self.flies)))
        
        # Schedule next update
        self.root.after(33, self.process_frame)
    
    def update_display(self, frame):
        """Update the display with current frame and flies"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw flies with wave brain influence
        for fly in self.flies:
            # Get average wave state from brain areas
            wave_state = np.mean([
                np.mean(list(fly.brain.wave_state['visual'])),
                np.mean(list(fly.brain.wave_state['motion'])),
                np.mean(list(fly.brain.wave_state['decision']))
            ])
            
            # Draw fly with wave influence
            self.fly_visuals.draw_fly(
                frame_rgb,
                fly.x,
                fly.y,
                fly.angle,
                size=self.config.fly_size,
                shade=0.7,
                wave_state=wave_state
            )
        
        # Convert to PhotoImage and display
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join()
        if hasattr(self, 'camera'):
            self.camera.release()

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create main window
    root = tk.Tk()
    
    try:
        # Create and run simulation
        app = WaveFlySimulation(root)
        
        # Setup cleanup
        def on_closing():
            app.cleanup()
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        messagebox.showerror("Error", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()