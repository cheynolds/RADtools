import numpy as np
import pyaudio
import sys
import time
import os


# Dynamically construct the path to the 'interfaces' folder in the user's home directory
home_dir = os.path.expanduser("~")  # Get the current user's home directory
interfaces_path = os.path.join(home_dir, "interfaces")  # Append 'interfaces' to the home directory path

try:
    # Check if the path exists and is a directory
    if not os.path.exists(interfaces_path):
        raise FileNotFoundError(f"Error: Path '{interfaces_path}' does not exist.")
    elif not os.path.isdir(interfaces_path):
        raise NotADirectoryError(f"Error: Path '{interfaces_path}' is not a directory.")

    # Safely append the path
    if interfaces_path not in sys.path:
        sys.path.append(interfaces_path)
        print(f"Successfully added '{interfaces_path}' to sys.path.")
    else:
        print(f"Path '{interfaces_path}' is already in sys.path.")
except (FileNotFoundError, NotADirectoryError) as e:
    print(e)
except Exception as e:
    print(f"An unexpected error occurred: {e}")

import pixels as ledInterface  # LED control for ReSpeaker

# Audio and Target Settings
TARGET_FREQ = 440  # Frequency in Hz
CHUNK = 32768  # Number of audio samples per frame, 
RATE = 48000  # Sampling rate in Hz
TOLERANCE = 1.5 # Tolerance range in Hz
GREEN_DURATION = 2.5  # Duration to keep green LED on after correct frequency

# LED Colors
leds = ledInterface.Pixels()


# Function to Scale LED Colors for Reduced Brightness
def scale_color(color, factor=0.25):
    return [int(c * factor) for c in color]


# LED Colors with 25% Brightness
ledCorrectColor = scale_color([0, 255, 0] * 3)      # Green for Correct Frequency
ledLowColor = scale_color([255, 0, 0] * 3)          # Red for Too Low
ledHighColor = scale_color([0, 0, 255] * 3)         # Blue for Too High
leds.write([0, 0, 0] * 3)  # Turn off LEDs initially

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Frequency Calculation with Hanning Window
def calculate_frequency(data, rate=48000):
    # Apply windowing
    window = np.hanning(len(data))
    windowed_data = data * window
    
    # Perform FFT and calculate magnitude spectrum
    fft_data = np.fft.fft(windowed_data)
    magnitude = np.abs(fft_data[:len(fft_data) // 2])
    
    # Find the frequency bin with the highest magnitude
    peak_bin = np.argmax(magnitude)
    
    # Convert the bin number to frequency
    freq = (peak_bin * rate) / len(data)
    return freq


# Main Tuner Function
def main():
    print(f"Tuning {TARGET_FREQ:.2f} Hz.")

    last_green_time = 0  # Time tracker for green LED duration
    green_led_timer_active = False  # Tracks if green LED timer is active

    try:
        while True:
            # Read audio data
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            padded_data = np.pad(data, (0, CHUNK), mode='constant')  # Zero-padding
            time.sleep(0.1)  # Small delay to reduce flickering
            # Calculate dominant frequency with windowing
            frequency = calculate_frequency(data)

            current_time = time.time()

            # Check if frequency is within the correct range
            if TARGET_FREQ - TOLERANCE <= frequency <= TARGET_FREQ + TOLERANCE:
                if not green_led_timer_active:
                    # Start the green LED timer
                    green_led_timer_active = True
                    last_green_time = current_time
                    print(f"Correct: {frequency:.2f} Hz")

                # Keep green LEDs on
                leds.write(ledCorrectColor)

            elif green_led_timer_active and (current_time - last_green_time < GREEN_DURATION):
                # Keep the green LEDs illuminated for the remainder of the 3 seconds
                leds.write(ledCorrectColor)
            else:
                # Reset green LED timer and resume high/low colors
                green_led_timer_active = False

                if frequency < TARGET_FREQ - TOLERANCE:
                    leds.write(ledLowColor)  # Red LEDs for too low
                    print(f"Too Low: {frequency:.2f} Hz")
                elif frequency > TARGET_FREQ + TOLERANCE:
                    leds.write(ledHighColor)  # Blue LEDs for too high
                    print(f"Too High: {frequency:.2f} Hz")

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        leds.write([0, 0, 0] * 3)  # Turn off LEDs

# Run the Tuner
if __name__ == "__main__":
    main()
