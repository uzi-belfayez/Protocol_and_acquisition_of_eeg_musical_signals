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
        
    def initialize(self):
        # 1. Start the TCP Listener in a separate thread
        self.running = True
        self.thread = threading.Thread(target=self.tcp_listener)
        self.thread.daemon = True
        self.thread.start()
        
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
        Expected format: "NOTE_60", "CHORD", "END"
        """
        try:
            tag = tag.strip()
            if tag.startswith("NOTE_"):
                # Example: NOTE_60 -> Base + 60
                midi_val = int(tag.split("_")[1])
                return self.BASE_LABEL + midi_val
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