import cv2
import mediapipe as mp
import numpy as np

# These are the MediaPipe landmark index numbers for each eye and the mouth.
# MediaPipe labels each of the 468 face points with a number from 0 to 467.
# These specific numbers correspond to the outer corners, top, and bottom of each eye.
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 39,  181, 0,   17,  269, 405]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    # Convert the landmark fractions (0.0 to 1.0) into actual pixel coordinates
    # by multiplying by the frame width (w) and height (h)
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    # A = vertical distance between top and bottom of eye (upper pair)
    # B = vertical distance between top and bottom of eye (lower pair)
    # C = horizontal distance from left corner to right corner of eye
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    
    # EAR formula: average vertical height divided by horizontal width
    # High EAR = eye open. Low EAR = eye closed.
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    A = np.linalg.norm(np.array(pts[2]) - np.array(pts[6]))
    B = np.linalg.norm(np.array(pts[3]) - np.array(pts[7]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[4]))
    return (A + B) / (2.0 * C)

def extract_features(landmarks, w, h):
    # Calculate both eye EARs and average them
    left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE,  w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ear = (left_ear + right_ear) / 2.0
    
    mar = mouth_aspect_ratio(landmarks, MOUTH, w, h)
    
    # Also flatten all 468 landmark x,y positions into a list
    # This gives the model more raw data to learn from
    coords = [v for i in range(468) for v in (landmarks[i].x, landmarks[i].y)]
    
    return ear, mar, coords

