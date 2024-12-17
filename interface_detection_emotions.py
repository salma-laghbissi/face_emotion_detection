










import cv2
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import face_recognition
from keras.models import load_model   
import collections


emotion_model = load_model('facialemotionmodel.h5')


known_face_encodings = []
known_face_ids = []
emotion_histories = collections.defaultdict(list)
next_person_id = 1  


cap = None
is_dark_theme = False


def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la caméra.")
        return
    update_frame()


def update_frame():
    global cap, image_label, known_face_encodings, known_face_ids, emotion_histories
    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_ids = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in match:
            first_match_index = match.index(True)
            name = known_face_ids[first_match_index]
        else:
            global next_person_id
            name = f"Person {next_person_id}"
            known_face_encodings.append(face_encoding)
            known_face_ids.append(name)
            emotion_histories[name]  
            next_person_id += 1

        face_ids.append(name)

        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    image_label.imgtk = imgtk
    image_label.configure(image=imgtk)

    image_label.after(10, update_frame)


def detect_emotion():
    global cap, emotion_histories
    if cap is None or not cap.isOpened():
        return

    ret, frame = cap.read()
    if not ret:
        return

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = None
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_ids[first_match_index]

        if name is None:
            continue

        emotion = detect_emotion_from_model(rgb_frame[top:bottom, left:right])
        if emotion:
            emotion_histories[name].append(emotion)
            update_emotion_history()


def detect_emotion_from_model(face_image):
    face_image = cv2.resize(face_image, (48, 48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    face_image = face_image.astype('float32') / 255
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)

    predictions = emotion_model.predict(face_image)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    predicted_emotion = emotion_labels[np.argmax(predictions)]

    return predicted_emotion


def update_emotion_history():
    global emotion_histories, emotion_history_text
    emotion_history_text.delete(1.0, tk.END)
    for person in sorted(emotion_histories.keys()):
        if emotion_histories[person]:
            history = f"{person}: {', '.join(emotion_histories[person])}\n"
            emotion_history_text.insert(tk.END, history)


def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    cv2.destroyAllWindows()


def show_about():
    messagebox.showinfo("À propos", "Application de détection des émotions faciales.\nDéveloppée avec Keras, OpenCV et Tkinter.")


def toggle_theme():
    global is_dark_theme
    is_dark_theme = not is_dark_theme
    bg_color = 'black' if is_dark_theme else 'white'
    fg_color = 'white' if is_dark_theme else 'black'
    root.configure(bg=bg_color)
    image_label.configure(bg=bg_color)
    emotion_history_text.configure(bg=bg_color, fg=fg_color)


def import_image():
    global emotion_histories, known_face_encodings, known_face_ids
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in match:
            first_match_index = match.index(True)
            name = known_face_ids[first_match_index]
        else:
            global next_person_id
            name = f"Person {next_person_id}"
            known_face_encodings.append(face_encoding)
            known_face_ids.append(name)
            emotion_histories[name]
            next_person_id += 1

        emotion = detect_emotion_from_model(image[top:bottom, left:right])
        if emotion:
            emotion_histories[name].append(emotion)
            update_emotion_history()


def show_help():
    help_message = (
        "Aide:\n"
        "1. Démarrer: Lance la caméra pour la détection des émotions.\n"
        "2. Détecter Émotions: Détecte les émotions des visages visibles dans la caméra.\n"
        "3. Arrêter: Stoppe la capture de la caméra.\n"
        "4. Importer: Charge une image depuis le disque et détecte les émotions.\n"
    )
    messagebox.showinfo("Aide", help_message)


root = tk.Tk()
root.title("Détection des émotions")
root.geometry("800x600")
root.configure(bg='white')


image_label = tk.Label(root, bg='black')
image_label.pack(side="top", padx=10, pady=10)


emotion_history_text = scrolledtext.ScrolledText(root, width=40, height=10, bg='white', fg='black')
emotion_history_text.pack(padx=10, pady=10)


start_button = tk.Button(root, text="Démarrer", command=start_camera, bg='lightblue', fg='black')
start_button.pack(side="left", padx=10, pady=10)

detect_button = tk.Button(root, text="Détecter Émotion", command=detect_emotion, bg='lightblue', fg='black')
detect_button.pack(side="left", padx=10, pady=10)

stop_button = tk.Button(root, text="Arrêter", command=stop_camera, bg='lightblue', fg='black')
stop_button.pack(side="left", padx=10, pady=10)

about_button = tk.Button(root, text="À propos", command=show_about, bg='lightblue', fg='black')
about_button.pack(side="right", padx=10, pady=10)

toggle_theme_button = tk.Button(root, text="Thème", command=toggle_theme, bg='lightblue', fg='black')
toggle_theme_button.pack(side="right", padx=10, pady=10)

import_button = tk.Button(root, text="Importer une Image", command=import_image, bg='lightblue', fg='black')
import_button.pack(side="right", padx=10, pady=10)

help_button = tk.Button(root, text="Aide", command=show_help, bg='lightblue', fg='black')
help_button.pack(side="right", padx=10, pady=10)

root.mainloop()


stop_camera()

































































































