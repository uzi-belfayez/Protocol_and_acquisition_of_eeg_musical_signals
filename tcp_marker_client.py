import pygame.midi
import socket
import time

# -----------------------------
# Configuration
# -----------------------------
OV_HOST = "127.0.0.1"
OV_PORT = 15361

# -----------------------------
# MIDI & Socket Setup
# -----------------------------
def init_system():
    pygame.midi.init()
    try:
        player = pygame.midi.Output(1) # Adjust ID if you have other MIDI devices
        player.set_instrument(0)       # Acoustic Grand Piano
    except Exception as e:
        print(f"MIDI Error: {e}")
        return None, None

    # Connect to OpenViBE
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"Connecting to OpenViBE at {OV_HOST}:{OV_PORT}...")
        sock.connect((OV_HOST, OV_PORT))
        print("Connected to OpenViBE!")
    except ConnectionRefusedError:
        print("Could not connect to OpenViBE. Is the scenario playing?")
        return None, None

    return player, sock

# -----------------------------
# Play & Send Logic
# -----------------------------
def play_note_with_marker(player, sock, note_name, midi_num, duration=0.5):
    """
    1. Sends marker to OpenViBE (Neuronal Sync)
    2. Plays the sound (Auditory Stimulus)
    """
    
    # 1. Send Marker (Format: NOTE_60\n)
    tag = f"NOTE_{midi_num}\n"
    try:
        sock.sendall(tag.encode('utf-8'))
        print(f"Sent marker: {tag.strip()}")
    except:
        print("Socket error sending marker")

    # 2. Play Sound
    # time.sleep(duration) #erroooooooooor
    player.note_on(midi_num, 100)
    time.sleep(duration)
    player.note_off(midi_num, 100)

# -----------------------------
# Helper Utilities (From your code)
# -----------------------------
NOTE_TO_MIDI = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11
}

def note_to_midi(note: str):
    if note[-1].isdigit():
        octave = int(note[-1])
        name = note[:-1]
    else:
        octave = 4
        name = note
    return 12 * (octave + 1) + NOTE_TO_MIDI[name]

# -----------------------------
# Main Experiment
# -----------------------------
if __name__ == "__main__":
    player, sock = init_system()

    if player and sock:
        try:
            # Let the EEG settle
            print("Starting in 2 seconds...")
            time.sleep(2)
            
            # Send Experiment Start Code
            sock.sendall(b"START\n")
            
            # Experiment Sequence
            sequence = [
    "C4", "C4", "C4", "C4", "C4",
    "A4", "A4", "A4", "A4", "A4",
    "D4", "D4", "D4", "D4", "D4",
    "C5", "C5", "C5", "C5", "C5",
    "F4", "F4", "F4", "F4", "F4",
    "B4", "B4", "B4", "B4", "B4",
    "E4", "E4", "E4", "E4", "E4",
    "G4", "G4", "G4", "G4", "G4",
    "B5", "B5", "B5", "B5", "B5",
    "F5", "F5", "F5", "F5", "F5",
    "A5", "A5", "A5", "A5", "A5",
    
    
]
            
            for note in sequence:
                midi_val = note_to_midi(note)
                print(f"Playing {note}...")
                play_note_with_marker(player, sock, note, midi_val, duration=0.15)
                
                # Inter-Stimulus Interval (ISI)
                # Randomize this in a real experiment to avoid anticipation!
                time.sleep(1) 

            # Send Experiment End Code
            sock.sendall(b"END\n")
            
        except KeyboardInterrupt:
            print("Stopped by user")
        finally:
            player.close()
            pygame.midi.quit()
            sock.close()
            print("Resources released.")