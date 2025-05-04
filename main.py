import os
import cv2
import face_recognition
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import numpy as np
import threading
import time
import logging
import configparser
import sqlite3
import queue  # Fixed import
from queue import Queue
import json

# Configuration setup
config = configparser.ConfigParser()
config.read('config.ini')

# Constants from config
try:
    TOLERANCE = float(config.get('DEFAULT', 'TOLERANCE', fallback=0.5))
    FRAME_REDUCTION = float(config.get('DEFAULT', 'FRAME_REDUCTION', fallback=0.25))
    ALERT_COOLDOWN = int(config.get('DEFAULT', 'ALERT_COOLDOWN', fallback=300))  # 5 minutes
    DATABASE_FILE = config.get('DEFAULT', 'DATABASE_FILE', fallback='face_database.db')
except Exception as e:
    print(f"Error reading config: {e}")
    TOLERANCE = 0.5
    FRAME_REDUCTION = 0.25
    ALERT_COOLDOWN = 300
    DATABASE_FILE = 'face_database.db'

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Create necessary directories
os.makedirs("known_faces", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drone_face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database setup
def init_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            encoding BLOB NOT NULL,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id INTEGER,
            detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (face_id) REFERENCES known_faces (id)
        )
    ''')
    conn.commit()
    conn.close()

init_database()

class FaceDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(DATABASE_FILE)
        self.lock = threading.Lock()
        
    def add_face(self, name, image_path, encoding):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO known_faces (name, image_path, encoding) VALUES (?, ?, ?)",
                (name, image_path, encoding.tobytes())
            )
            self.conn.commit()
            return cursor.lastrowid
    
    def get_all_faces(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, encoding FROM known_faces")
        results = cursor.fetchall()
        names = []
        encodings = []
        for name, encoding_blob in results:
            names.append(name)
            encodings.append(np.frombuffer(encoding_blob, dtype=np.float64))
        return encodings, names
    
    def log_detection(self, face_id):
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO detections (face_id) VALUES (?)",
                (face_id,)
            )
            self.conn.commit()
    
    def __del__(self):
        self.conn.close()

face_db = FaceDatabase()

class EmailService:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.cooldowns = {}
        
    def send_email_alert(self, name):
        try:
            # Check cooldown
            current_time = time.time()
            last_alert = self.cooldowns.get(name, 0)
            if current_time - last_alert < ALERT_COOLDOWN:
                logger.info(f"Alert for {name} is in cooldown period")
                return
            
            # Get email settings from config
            sender_email = self.config.get('EMAIL', 'SENDER_EMAIL')
            receiver_email = self.config.get('EMAIL', 'RECEIVER_EMAIL')
            password = self.config.get('EMAIL', 'PASSWORD')
            
            msg = MIMEText(f"🚨 ALERT: {name} was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            msg['Subject'] = f"Drone Alert: {name} Detected"
            msg['From'] = sender_email
            msg['To'] = receiver_email

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, password)
                server.send_message(msg)
                logger.info(f"Email sent for {name}")
                self.cooldowns[name] = current_time
                
        except Exception as e:
            logger.error(f"Email failed: {e}")

email_service = EmailService()

def image_quality_check(image):
    """Basic image quality assessment"""
    # Check if image is too dark
    if np.mean(image) < 30:
        return False, "Image too dark"
    
    # Check if image is blurry
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    if fm < 100:
        return False, "Image too blurry"
    
    return True, "Image quality OK"

def draw_face_box(frame, name, top, right, bottom, left):
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), FONT, 0.7, (255, 255, 255), 2)

class CameraThread(threading.Thread):
    def __init__(self, queue, mode='recognition'):
        threading.Thread.__init__(self)
        self.queue = queue
        self.mode = mode
        self.running = True
        self.known_encodings, self.known_names = face_db.get_all_faces()
        self.alerted = set()
        
    def run(self):
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            logger.error("Cannot open camera")
            self.queue.put(('error', "Cannot open camera"))
            return
        
        try:
            while self.running:
                ret, frame = video.read()
                if not ret:
                    logger.warning("Failed to grab frame")
                    continue
                
                small_frame = cv2.resize(frame, (0, 0), fx=FRAME_REDUCTION, fy=FRAME_REDUCTION)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Fixed color conversion

                face_locations = face_recognition.face_locations(rgb_small)
                
                try:
                    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                except Exception as e:
                    logger.error(f"Face encoding error: {e}")
                    continue

                current_names = []

                for encoding, location in zip(face_encodings, face_locations):
                    name = "Unknown"
                    matches = face_recognition.compare_faces(self.known_encodings, encoding, tolerance=TOLERANCE)
                    face_distances = face_recognition.face_distance(self.known_encodings, encoding)

                    if matches and len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_names[best_match_index]
                            if name not in self.alerted and self.mode == 'recognition':
                                email_thread = threading.Thread(
                                    target=email_service.send_email_alert,
                                    args=(name,)
                                )
                                email_thread.start()
                                self.alerted.add(name)

                    current_names.append(name)
                    top, right, bottom, left = [int(v / FRAME_REDUCTION) for v in location]
                    draw_face_box(frame, name, top, right, bottom, left)

                # Add sidebar
                sidebar_width = 200
                overlay = frame.copy()
                cv2.rectangle(overlay, (frame.shape[1] - sidebar_width, 0), 
                             (frame.shape[1], frame.shape[0]), (30, 30, 30), -1)
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                y_offset = 30
                for name in current_names:
                    label = f"• {name}"
                    cv2.putText(frame, label, (frame.shape[1] - sidebar_width + 10, y_offset), 
                                FONT, 0.7, (255, 255, 255), 2)
                    y_offset += 30

                # Put frame in queue for GUI
                self.queue.put(('frame', frame))
                
                # Small delay to prevent high CPU usage
                time.sleep(0.03)
                
        except Exception as e:
            logger.error(f"Camera thread error: {e}")
            self.queue.put(('error', str(e)))
        finally:
            video.release()
            logger.info("Camera released")

    def stop(self):
        self.running = False

def start_recognition():
    def update_gui():
        while True:
            try:
                msg_type, data = camera_queue.get_nowait()
                if msg_type == 'frame':
                    # Convert the image to PhotoImage
                    img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Update the label
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    break
            except queue.Empty:  # Fixed exception handling
                break
        if hasattr(start_recognition, 'camera_thread') and start_recognition.camera_thread.is_alive():
            root.after(50, update_gui)
        else:
            stop_camera()

    def stop_camera():
        if hasattr(start_recognition, 'camera_thread'):
            start_recognition.camera_thread.stop()
            start_recognition.camera_thread.join()
        video_label.config(image=None)
        video_label.image = None
        control_frame.pack()
        if 'video_window' in locals() and video_window.winfo_exists():
            video_window.destroy()

    # Create a new window for video display
    video_window = tk.Toplevel(root)
    video_window.title("Face Recognition")
    video_window.protocol("WM_DELETE_WINDOW", stop_camera)

    # Video display
    video_label = tk.Label(video_window)
    video_label.pack()

    # Control frame
    control_frame = tk.Frame(video_window)
    control_frame.pack(pady=10)

    stop_button = tk.Button(control_frame, text="Stop", command=stop_camera)
    stop_button.pack()

    # Start camera thread
    camera_queue = Queue()
    camera_thread = CameraThread(camera_queue, mode='recognition')
    camera_thread.start()
    start_recognition.camera_thread = camera_thread
    start_recognition.camera_queue = camera_queue

    # Hide main window controls
    control_frame.pack_forget()

    # Start GUI update loop
    update_gui()

def live_view_only():
    def update_gui():
        while True:
            try:
                msg_type, data = camera_queue.get_nowait()
                if msg_type == 'frame':
                    # Convert the image to PhotoImage
                    img = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Update the label
                    video_label.imgtk = imgtk
                    video_label.configure(image=imgtk)
                elif msg_type == 'error':
                    messagebox.showerror("Error", data)
                    break
            except queue.Empty:  # Fixed exception handling
                break
        if hasattr(live_view_only, 'camera_thread') and live_view_only.camera_thread.is_alive():
            root.after(50, update_gui)
        else:
            stop_camera()

    def stop_camera():
        if hasattr(live_view_only, 'camera_thread'):
            live_view_only.camera_thread.stop()
            live_view_only.camera_thread.join()
        video_label.config(image=None)
        video_label.image = None
        control_frame.pack()
        if 'video_window' in locals() and video_window.winfo_exists():
            video_window.destroy()

    # Create a new window for video display
    video_window = tk.Toplevel(root)
    video_window.title("Live View")
    video_window.protocol("WM_DELETE_WINDOW", stop_camera)

    # Video display
    video_label = tk.Label(video_window)
    video_label.pack()

    # Control frame
    control_frame = tk.Frame(video_window)
    control_frame.pack(pady=10)

    stop_button = tk.Button(control_frame, text="Stop", command=stop_camera)
    stop_button.pack()

    # Start camera thread
    camera_queue = Queue()
    camera_thread = CameraThread(camera_queue, mode='view_only')
    camera_thread.start()
    live_view_only.camera_thread = camera_thread
    live_view_only.camera_queue = camera_queue

    # Hide main window controls
    control_frame.pack_forget()

    # Start GUI update loop
    update_gui()

def add_new_face():
    name = simpledialog.askstring("New Face", "Enter the name of the person:")
    if not name:
        return

    video = cv2.VideoCapture(0)
    if not video.isOpened():
        messagebox.showerror("Error", "Cannot open camera")
        return

    capture_window = tk.Toplevel(root)
    capture_window.title("Add New Face - Press SPACE to capture, ESC to cancel")
    capture_window.focus_force()
    
    preview_label = tk.Label(capture_window)
    preview_label.pack()

    def update_capture_preview():
        ret, frame = video.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            preview_label.imgtk = imgtk
            preview_label.configure(image=imgtk)
            preview_label.image = imgtk
        
        if add_new_face.capturing:
            capture_window.after(30, update_capture_preview)
        else:
            video.release()
            capture_window.destroy()

    def on_key_press(event):
        if event.keysym == 'Escape':
            add_new_face.capturing = False
        elif event.keysym == 'space':
            ret, frame = video.read()
            if ret:
                quality_ok, quality_msg = image_quality_check(frame)
                if not quality_ok:
                    messagebox.showwarning("Quality Issue", f"Cannot save: {quality_msg}")
                    return
                
                image_path = os.path.join("known_faces", f"{name}.jpg")
                cv2.imwrite(image_path, frame)
                
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_image)
                if encodings:
                    face_db.add_face(name, image_path, encodings[0])
                    messagebox.showinfo("Saved", f"Face saved as {image_path}")
                else:
                    messagebox.showerror("Error", "No face detected in the captured image")
                    os.remove(image_path)
                
                add_new_face.capturing = False

    capture_window.bind('<Key>', on_key_press)
    add_new_face.capturing = True
    update_capture_preview()

# GUI Setup
root = tk.Tk()
root.title("🚁 Drone Face Recognition System")
root.geometry("400x350")
root.configure(bg="#1e1e2e")

if not os.path.exists('config.ini'):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'TOLERANCE': '0.5',
        'FRAME_REDUCTION': '0.25',
        'ALERT_COOLDOWN': '300'
    }
    config['EMAIL'] = {
        'SENDER_EMAIL': 'your_email@gmail.com',
        'RECEIVER_EMAIL': 'receiver_email@gmail.com',
        'PASSWORD': 'your_app_password'
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    messagebox.showinfo("Configuration", "A new config.ini file has been created. Please update your email settings.")

# Main GUI
tk.Label(root, text="Drone Face Recognition System", font=("Helvetica", 16, "bold"), fg="white", bg="#1e1e2e").pack(pady=15)
tk.Button(root, text="🟢 Start Face Recognition", command=start_recognition, width=30, bg="#2ecc71", fg="white", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="🔍 View Live Feed", command=live_view_only, width=30, bg="#9b59b6", fg="white", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="➕ Add New Face", command=add_new_face, width=30, bg="#3498db", fg="white", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="⚙️ Settings", command=lambda: os.startfile('config.ini'), width=30, bg="#f39c12", fg="white", font=("Arial", 12)).pack(pady=10)
tk.Button(root, text="❌ Exit", command=root.quit, width=30, bg="#e74c3c", fg="white", font=("Arial", 12)).pack(pady=10)

root.mainloop()