import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from fer import FER

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection")

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            return

        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Initialize the emotion detector
        self.detector = FER()

        # Create a label to display the camera feed
        self.label = Label(root)
        self.label.pack()

        # Initialize thread lock
        self.lock = threading.Lock()

        # Start the video thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        # Close the camera when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def video_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detect emotion in the face region
                emotions = self.detector.detect_emotions(roi_color)
                if emotions:
                    # Get the most likely emotion
                    dominant_emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                    emotion_text = f"Emotion: {dominant_emotion}"
                    
                    # Draw a rectangle around the face and put the emotion text
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Convert the frame to an image format Tkinter can use
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            # Update the label with the new frame
            self.update_image(image)

    def update_image(self, image):
        # Update the label with the new frame in a thread-safe way
        with self.lock:
            self.label.config(image=image)
            self.label.image = image

    def on_close(self):
        # Release the camera and destroy the window
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()
