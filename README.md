Created by- Satish Sharma
🚁 Drone Face Recognition
A cutting-edge tool that enables a drone to spot and identify missing individuals using live facial recognition. This system combines real-time drone control with advanced deep learning methods for face detection and identification.

🔍 Key Features
Database-driven recognition: Stores images and information of missing individuals in a .csv file for database-driven identification.

Face extraction & embedding: Uses MTCNN to detect faces, and Google's FaceNet (Inception ResNet V1) to generate facial embeddings, which are stored in a known_faces.pt PyTorch file.

Live recognition: The drone’s camera feed is continuously matched against the saved embeddings in real time.

User-controlled navigation: Enables manual flight control (e.g., with Pygame or Tello SDK), overlaying bounding boxes and identity labels on recognized faces.

⚙️ How It Works
Database setup – Add or update a CSV with missing persons' IDs, names, and image paths.

Train the model – Run training.py to process images, detect faces, learn embeddings, and save them.

Start recognition – Launch main.py (or drone_deep_learning_facenet.py) to stream video from the drone, detect and identify faces live, and overlay identity information.
