import cv2
import mediapipe as mp
import numpy as np

# A little helper function to get the angle of a joint
def calculate_angle(a, b, c):
    # Convert points to numpy arrays
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    # Get the angle in radians and then convert to degrees
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # The angle can sometimes be > 180, so we correct for it
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Path to my video file
VIDEO_PATH = "lo.mp4" 
cap = cv2.VideoCapture(VIDEO_PATH)

# Setup mediapipe instance
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# My pushup counter variables
pushup_count = 0
stage = None # This will be either "up" or "down"

# Main loop to go through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break

    # MediaPipe works with RGB, but OpenCV uses BGR, so we need to convert.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Let's get the pose detection results
    results = pose.process(image)
    
    # Convert back to BGR so we can display it with OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # --- I was playing with some anti-cheat stuff here, but it's not really needed. ---
    # # Timestamp anomaly check
    # timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    # # Frame rate consistency check
    # # Scene change detection
    # --- End of extra stuff ---

    # We need to extract the landmarks (i.e., the joints)
    try:
        landmarks = results.pose_landmarks.landmark
        
        # Get coordinates for the right arm. You could use the left arm too.
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Now, calculate the elbow angle
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # --- The actual push-up counting logic ---
        if angle > 160:
            stage = "up"
        # If we are in the "up" stage and the angle goes below 90, we've completed one rep.
        if angle < 90 and stage == 'up':
            stage = "down"
            pushup_count += 1
            print(f"Count: {pushup_count}") # Also print to console
            
        # Display the angle and count on the screen
        cv2.putText(image, f'Angle: {int(angle)}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f'Push-Ups: {pushup_count}', (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    except:
        # This will just pass if no pose is detected in a frame
        pass
    
    # Draw the landmarks on the person
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)                       
    
    cv2.imshow('Push-Up Counter', image)

    # Let me quit by pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up everything when we're done
cap.release()
cv2.destroyAllWindows()