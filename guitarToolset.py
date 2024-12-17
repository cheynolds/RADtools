import queue
import numpy as np
import pyaudio
import sys
import time
import threading
import RPi.GPIO as GPIO  # For button input
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
targetFreq = 440  # String frequency in Hz
chunkSize = 4096  # Larger buffer size for reduced skipping
chunkSizeTUNE = 32768 #Tuning chunk size to increase resolution
sampleRate = 48000  # Sampling rate in Hz
tolerance = 3  # Tolerance range in Hz
greenDuration = 0.5  # Duration to keep green LED on after correct frequency

# LED Colors (Reduced Brightness by 50%)
def scaleColor(color, factor=0.25):
    return [int(c * factor) for c in color]

leds = ledInterface.Pixels()
ledCorrectColor = scaleColor([0, 255, 0] * 3)       # Green
ledLowColor = scaleColor([255, 0, 0] * 3)           # Red
ledHighColor = scaleColor([0, 0, 255] * 3)          # Blue
ledOverdriveColor = scaleColor([255, 255, 0] * 3)   # Yellow for Distortion Mode
ledFlangerColor = scaleColor([0, 0, 255] * 3)       # Blue for Flanger Mode
ledTremoloColor = scaleColor([255, 105, 180] * 3)   # Pink for Tremolo Mode
ledFuzzColor = scaleColor([128, 0, 0] * 3)          # Dark Red for Fuzz Mode
leds.write([0, 0, 0] * 3)  # Turn off LEDs initially

# GPIO Setup
buttonPin = 17  # Adjust to your button's GPIO pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(buttonPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Initialize PyAudio
pAudio = pyaudio.PyAudio()

# Function to find Scarlett device
def getScarlettDeviceIndex():
    """
    Searches for the Scarlett audio interface among the available devices.
    """
    for i in range(pAudio.get_device_count()):
        device = pAudio.get_device_info_by_index(i)
        if "Scarlett" in device['name']:
            print(f"Found Scarlett device: {device['name']} (Index: {i})")
            return i
    print("Scarlett device not found.")
    return None

scarlettIndex = getScarlettDeviceIndex()
if scarlettIndex is None:
    print("Exiting program: Scarlett device not detected.")
    sys.exit()

# Open audio input and output streams
inputStream = pAudio.open(format=pyaudio.paInt16, channels=1, rate=sampleRate, input=True, 
                          input_device_index=scarlettIndex, frames_per_buffer=chunkSize)
outputStream = pAudio.open(format=pyaudio.paInt16, channels=1, rate=sampleRate, output=True, 
                           frames_per_buffer=chunkSize)


# Global State for Effect
currentEffect = None

# Effect Functions
def applyOverdrive(signal, gain=2.0):
    """
    Applies an overdrive effect to the input signal by amplifying and clipping it.

    Parameters:
        signal (numpy.ndarray): The input audio signal.
        gain (float): Gain factor to amplify the signal (default: 2.0).

    Returns:
        numpy.ndarray: The overdrive-processed audio signal (int16 format).
    """
    # Amplify the input signal
    amplifiedSignal = signal * gain

    # Clip the amplified signal to int16 range
    clippedSignal = np.clip(amplifiedSignal, -32768, 32767)

    # Return the processed signal as int16
    return clippedSignal.astype(np.int16)


def applyFlanger(signal, sampleRate, flangerLFO, maxDelayMs=3):
    """
    Applies a flanger effect to the input signal using an LFO (Low-Frequency Oscillator).

    Parameters:
        signal (numpy.ndarray): The input audio signal.
        sampleRate (int): The sampling rate of the signal.
        flangerLFO (numpy.ndarray): Precomputed LFO waveform to control delay.
        maxDelayMs (int): Maximum delay in milliseconds for the flanger effect (default: 3 ms).

    Returns:
        numpy.ndarray: The flanger-processed audio signal (int16 format).
    """
    # Calculate max delay in samples
    maxDelaySamples = int(maxDelayMs * sampleRate / 1000)
    numSamples = len(signal)
    
    # Ensure flangerLFO is scaled correctly to delay range
    delaySamples = (flangerLFO * maxDelaySamples / 2).astype(np.int32)
    outputSignal = np.zeros_like(signal, dtype=np.float32)

    for i in range(numSamples):
        delay = delaySamples[i]
        if i - delay >= 0:  # Ensure no negative indexing
            outputSignal[i] = 0.5 * signal[i] + 0.5 * signal[i - delay]
        else:
            outputSignal[i] = signal[i]  # Pass input if delay goes out of bounds

    # Clip the output to int16 range and return
    return np.clip(outputSignal, -32768, 32767).astype(np.int16)




def applyTremolo(signal, sampleRate, modulationFrequency=5):
    """
    Applies a tremolo effect to the input signal by modulating its amplitude
    with a low-frequency oscillator (LFO).

    Parameters:
        signal (numpy.ndarray): The input audio signal.
        sampleRate (int): The sampling rate of the signal.
        modulationFrequency (float): Frequency of the amplitude modulation in Hz (default: 5 Hz).

    Returns:
        numpy.ndarray: The tremolo-processed audio signal (int16 format).
    """
    # Create a time vector based on the input signal length
    timeVector = np.arange(len(signal)) / sampleRate
    
    # Generate the LFO (Low-Frequency Oscillator) for amplitude modulation
    lfo = 0.5 * (1 + np.sin(2 * np.pi * modulationFrequency * timeVector))
    
    # Apply the LFO to modulate the signal amplitude
    outputSignal = signal * lfo

    # Return the processed signal clipped to int16 range
    return np.clip(outputSignal, -32768, 32767).astype(np.int16)


def applyFuzz(signal, gain=2.0, distortionCoefficient=10):
    """
    Applies a fuzz distortion effect to the input signal by amplifying and shaping its waveform.

    Parameters:
        signal (numpy.ndarray): The input audio signal.
        gain (float): Gain factor to amplify the signal (default: 2.0).
        distortionCoefficient (float): Controls the amount of distortion applied (default: 10).

    Returns:
        numpy.ndarray: The fuzz-distorted audio signal (int16 format).
    """
    # Amplify the input signal
    amplifiedSignal = signal * gain

    # Small constant to avoid division by zero
    epsilon = 1e-10

    # Apply the fuzz distortion function
    distortedSignal = np.sign(amplifiedSignal) * (
        1 - np.exp(-distortionCoefficient * (amplifiedSignal ** 2) / (np.abs(amplifiedSignal) + epsilon))
    )

    # Clip the output signal to int16 range and return as int16
    return np.clip(distortedSignal, -32768, 32767).astype(np.int16)


def calculateFrequency(signal):
    """
    Calculates the dominant frequency in the audio signal using FFT.
    """
    window = np.hanning(len(signal))
    windowedSignal = signal * window
    fftSignal = np.fft.rfft(windowedSignal)
    freqs = np.fft.rfftfreq(len(windowedSignal), 1.0 / sampleRate)
    magnitude = np.abs(fftSignal)
    return freqs[np.argmax(magnitude)]

def tuner():
    print("Tuning 440Hz. Listening...")
    lastGreenTime = 0
    try:
        while True:
            if GPIO.input(buttonPin) == GPIO.LOW:
                print("Exiting tuner mode...")
                break

            signal = np.frombuffer(inputStream.read(chunkSizeTUNE, exception_on_overflow=False), dtype=np.int16)
            frequency = calculateFrequency(signal)

            if targetFreq - tolerance <= frequency <= targetFreq + tolerance:
                leds.write(ledCorrectColor)
                lastGreenTime = time.time()
                print(f"Correct: {frequency:.2f} Hz")
            elif time.time() - lastGreenTime < greenDuration:
                leds.write(ledCorrectColor)
            elif frequency < targetFreq - tolerance:
                leds.write(ledLowColor)
                print(f"Too Low: {frequency:.2f} Hz")
            elif frequency > targetFreq + tolerance:
                leds.write(ledHighColor)
                print(f"Too High: {frequency:.2f} Hz")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        leds.write([0, 0, 0] * 3)

def processAudio(signal):
    global currentEffect
    if currentEffect == 'overdrive':
        leds.write(ledOverdriveColor)
        return applyOverdrive(signal)
    elif currentEffect == 'flanger':
        leds.write(ledFlangerColor)
        return applyFlanger(signal)
    elif currentEffect == 'tremolo':
        leds.write(ledTremoloColor)
        return applyTremolo(signal)
    elif currentEffect == 'fuzz':
        leds.write(ledFuzzColor)
        return applyFuzz(signal)
    else:
        leds.write([0, 0, 0] * 3)
        return signal

# Shared audio queue between input and output threads
audioQueue = queue.Queue()

def inputThread(inputStream, chunkSize):
    """
    Reads audio data from the input stream, processes it, and puts it into the queue.
    """
    try:
        while True:
            # Read raw audio data
            signal = np.frombuffer(inputStream.read(chunkSize, exception_on_overflow=False), dtype=np.int16)
            
            # Apply the current audio effect
            processedSignal = processAudio(signal)
            
            # Add processed signal to the queue
            audioQueue.put(processedSignal.tobytes())
    except KeyboardInterrupt:
        print("Input thread stopped.")
        audioQueue.put(None)  # Signal output thread to exit

def outputThread(outputStream):
    """
    Writes audio data from the queue to the output stream.
    """
    try:
        while True:
            # Get processed audio data from the queue
            data = audioQueue.get()
            if data is None:  # Exit signal
                break
            outputStream.write(data)
    except KeyboardInterrupt:
        print("Output thread stopped.")

def audioStreaming():
    """
    Starts input and output threads for real-time audio processing.
    """
    print("Audio streaming started. Use the button to select effects.")
    
    # Create and start threads
    inputThreadInstance = threading.Thread(target=inputThread, args=(inputStream, chunkSize), daemon=True)
    outputThreadInstance = threading.Thread(target=outputThread, args=(outputStream,), daemon=True)
    
    inputThreadInstance.start()
    outputThreadInstance.start()

    try:
        # Keep main thread alive while input and output threads are running
        inputThreadInstance.join()
        outputThreadInstance.join()
    except KeyboardInterrupt:
        print("Exiting audio streaming...")
        audioQueue.put(None)  # Signal output thread to exit
    finally:
        # Clean up resources
        leds.write([0, 0, 0] * 3)
        inputStream.stop_stream()
        outputStream.stop_stream()
        inputStream.close()
        outputStream.close()


def displayMenu():
    print("""
***********************************************
*      1 Press: Return to Menu                *
*      2 Presses: Tuner Mode                  *
*      3 Presses: Overdrive Mode              *
*      4 Presses: Tremolo Mode                *
*      5 Presses: Flanger Mode                *
*      6 Presses: Fuzz Mode                   *
***********************************************
""")

def menuHandler():
    global currentEffect
    buttonPressCount = 0
    buttonLastPressed = time.time()
    try:
        while True:
            if GPIO.input(buttonPin) == GPIO.LOW and time.time() - buttonLastPressed > 0.2:
                buttonPressCount += 1
                buttonLastPressed = time.time()
            if buttonPressCount > 0 and time.time() - buttonLastPressed > 1.0:
                if buttonPressCount == 1:
                    displayMenu()
                    currentEffect = None
                elif buttonPressCount == 2:
                    print("Tuner mode selected.")
                    tuner()
                elif buttonPressCount == 3:
                    print("Overdrive mode selected.")
                    currentEffect = 'overdrive'
                elif buttonPressCount == 4:
                    print("Tremolo mode selected.")
                    currentEffect = 'tremolo'
                elif buttonPressCount == 5:
                    print("Flanger mode selected.")
                    currentEffect = 'flanger'
                elif buttonPressCount == 6:
                    print("Fuzz mode selected.")
                    currentEffect = 'fuzz'
                buttonPressCount = 0
    except KeyboardInterrupt:
        print("Exiting menu handler...")

def main():
    displayMenu()
    audioThread = threading.Thread(target=audioStreaming, daemon=True)
    audioThread.start()
    menuHandler()

if __name__ == "__main__":
    main()
