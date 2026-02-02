import socket
import threading
import select
import time
from queue import Queue

# OpenViBE specific classes are injected by the environment, 
# but we import them here to suppress IDE warnings if you were editing outside OpenViBE.
# In the actual OpenViBE execution, these are already available.
try:
    OVBox
except NameError:
    class OVBox: pass
    class OVStimulationHeader: pass
    class OVStimulationSet: pass
    class OVStimulation: pass
    class OVStimulationEnd: pass

class TCPMarkerBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)
        self.signalHeader = None
        self.cmd_queue = Queue() # Thread-safe queue to store incoming markers
        self.server_socket = None
        self.running = False
        self.thread = None
        
        # --- CONFIGURATION ---
        self.HOST = "127.0.0.1"
        self.PORT = 15361
        self.BASE_LABEL = 33000 # OVTK_StimulationId_Label_00
        self.FILE_START_BASE = 34000
        self.FILE_END_BASE = 35000
        self.FILE_MAP_PATH = r"C:\Users\rayen\eeg\file_marker_map.csv"
        self.file_code_map = {}
        self.file_code_next = 0
        
    def initialize(self):
        # 1. Start the TCP Listener in a separate thread
        self.running = True
        self.thread = threading.Thread(target=self.tcp_listener)
        self.thread.daemon = True
        self.thread.start()

        self.load_file_map()

        # 2. Send the Stimulation Header (Required for the stream to start properly)
        # We assume the stream starts at t=0
        header = OVStimulationHeader(0., 0.)
        self.output[0].append(header)
        
        print(f"[TCP Box] Listening on {self.HOST}:{self.PORT}")
        return

    def process(self):
        # This function is called repeatedly by OpenViBE's clock
        
        # 1. Check if we have pending markers in the queue
        if not self.cmd_queue.empty():
            
            # Get current OpenViBE time
            current_time = self.getCurrentTime()
            
            # Create a Stimulation Set for this moment
            # We create a tiny window for the chunk
            stim_set = OVStimulationSet(current_time, current_time + 0.01)
            
            while not self.cmd_queue.empty():
                tag = self.cmd_queue.get()
                stim_code = self.parse_tag(tag)
                
                if stim_code:
                    # Create the stimulation object
                    # OVStimulation(Identifier, Date, Duration)
                    stim = OVStimulation(stim_code, current_time, 0.0)
                    stim_set.append(stim)
                    print(f"[TCP Box] Sent Stimulation: {stim_code} at {current_time}")
            
            # Send the set to Output 0
            self.output[0].append(stim_set)
            
        return

    def uninitialize(self):
        # Clean up the socket and thread
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        return

    def parse_tag(self, tag):
        """
        Converts string tags to OpenViBE Integer codes.
        Expected format: numeric codes, "NOTE_60", "CHORD", "END"
        """
        try:
            tag = tag.strip()
            if tag.isdigit():
                return int(tag)
            if tag.startswith("NOTE_"):
                # Example: NOTE_60 -> Base + 60
                midi_val = int(tag.split("_")[1])
                return self.BASE_LABEL + midi_val
            elif tag.startswith("START_"):
                name = tag[6:].strip().upper()
                start_code, _ = self.get_file_codes(name)
                return start_code
            elif tag.startswith("END_"):
                name = tag[4:].strip().upper()
                _, end_code = self.get_file_codes(name)
                return end_code
            elif tag == "START":
                return 32769 # OVTK_StimulationId_ExperimentStart
            elif tag == "END":
                return 32770 # OVTK_StimulationId_ExperimentStop
            elif tag == "CHORD":
                return 33200 # Custom code for chords
            else:
                return 0 # Unknown
        except:
            return 0

    def load_file_map(self):
        try:
            with open(self.FILE_MAP_PATH, "r", encoding="ascii") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("file_name"):
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 3:
                        continue
                    name, start_code, end_code = parts[0], int(parts[1]), int(parts[2])
                    self.file_code_map[name] = (start_code, end_code)
                if self.file_code_map:
                    used = [pair[0] - self.FILE_START_BASE for pair in self.file_code_map.values()]
                    self.file_code_next = max(used) + 1
        except Exception:
            pass

    def persist_file_code(self, name, start_code, end_code):
        try:
            new_file = False
            try:
                with open(self.FILE_MAP_PATH, "r", encoding="ascii"):
                    pass
            except FileNotFoundError:
                new_file = True
            with open(self.FILE_MAP_PATH, "a", encoding="ascii") as f:
                if new_file:
                    f.write("file_name,start_code,end_code\n")
                f.write(f"{name},{start_code},{end_code}\n")
        except Exception:
            pass

    def get_file_codes(self, name):
        if name in self.file_code_map:
            return self.file_code_map[name]
        idx = self.file_code_next
        self.file_code_next += 1
        start_code = self.FILE_START_BASE + idx
        end_code = self.FILE_END_BASE + idx
        self.file_code_map[name] = (start_code, end_code)
        self.persist_file_code(name, start_code, end_code)
        return start_code, end_code

    def tcp_listener(self):
        """Background thread to handle TCP connections"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.server_socket.bind((self.HOST, self.PORT))
            self.server_socket.listen(1)
            self.server_socket.settimeout(0.5) # Non-blocking accept
        except Exception as e:
            print(f"[TCP Box] Error binding: {e}")
            return

        while self.running:
            try:
                client, addr = self.server_socket.accept()
                print(f"[TCP Box] Connected to {addr}")
                
                with client:
                    while self.running:
                        data = client.recv(1024)
                        if not data: break
                        
                        # Handle multiple commands in one packet (split by newline)
                        messages = data.decode('utf-8').split('\n')
                        for msg in messages:
                            if msg:
                                self.cmd_queue.put(msg)
            except socket.timeout:
                continue # Loop back and check self.running
            except Exception as e:
                pass

# !!! CRITICAL !!!
# This line instantiates the box so OpenViBE can find it.
box = TCPMarkerBox()
