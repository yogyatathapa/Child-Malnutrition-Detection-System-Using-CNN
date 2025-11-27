import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("malnrtri.keras")  # Updated model file
class_names = ["Malnourished", "Normal", "Overmalnourished"]

# Global variables
img_display = None
img_path = None

# ----------------- Functions -----------------
def select_image():
    global img_display, img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if img_path:
        img = Image.open(img_path)
        img = img.resize((400, 300))
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)
        result_label.config(text="Result: ")

def detect_image():
    if not img_path:
        result_label.config(text="Result: Please select an image first!")
        return
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    result_label.config(text=f"Result: {predicted_class}")

def delete_image():
    global img_display, img_path
    img_display = None
    img_path = None
    image_label.config(image='')
    result_label.config(text="Result: ")

def clear_result():
    result_label.config(text="Result: ")

# ----------------- Button hover effect -----------------
def on_enter(e):
    if e.widget in [select_btn, detect_btn]:
        e.widget['bg'] = 'green'
    else:
        e.widget['bg'] = 'red'

def on_leave(e):
    e.widget['bg'] = 'white'

# ----------------- Tkinter Window -----------------
root = tk.Tk()
root.title("Child Malnutrition Detection System")
root.configure(bg="white")
root.geometry("800x700")
root.minsize(800, 700)
root.resizable(True, True)

# ----------------- Header -----------------
header_frame = tk.Frame(root, bg="#2E3EC7", height=80)
header_frame.pack(fill="x", padx=0, pady=0)

# Logo
logo_img = Image.open("M.png")
logo_img = logo_img.resize((90, 90)) 
logo_display = ImageTk.PhotoImage(logo_img)
logo_label = tk.Label(header_frame, image=logo_display, bg="#2E3EC7")
logo_label.pack(side="left", padx=15, pady=10) 

header_label = tk.Label(header_frame, text="Child Malnutrition Detection System",
                        bg="#2E3EC7", fg="white", font=("Arial", 20, "bold"))
header_label.place(relx=0.5, rely=0.5, anchor="center")

# ----------------- Buttons Box -----------------
button_box = tk.Frame(root, bg="#2E3EC7", pady=10)
button_box.pack(pady=10)

select_btn = tk.Button(button_box, text="Select Image", width=15, bg="white", fg="black", command=select_image)
select_btn.grid(row=0, column=0, padx=10)
select_btn.bind("<Enter>", on_enter)
select_btn.bind("<Leave>", on_leave)

detect_btn = tk.Button(button_box, text="Detect", width=15, bg="white", fg="black", command=detect_image)
detect_btn.grid(row=0, column=1, padx=10)
detect_btn.bind("<Enter>", on_enter)
detect_btn.bind("<Leave>", on_leave)

delete_btn = tk.Button(button_box, text="Delete", width=15, bg="white", fg="black", command=delete_image)
delete_btn.grid(row=0, column=2, padx=10)
delete_btn.bind("<Enter>", on_enter)
delete_btn.bind("<Leave>", on_leave)

# ----------------- Image Preview -----------------
image_label = tk.Label(root, bg="white")
image_label.pack(pady=10)

# ----------------- Result Area -----------------
result_frame = tk.Frame(root, bg="#2E3EC7", pady=10)
result_frame.pack(pady=5)

result_label = tk.Label(result_frame, text="Result: ", bg="#2E3EC7", fg="white", font=("Arial", 14))
result_label.pack(pady=5)

clear_btn = tk.Button(result_frame, text="Clear", width=10, bg="white", fg="black", command=clear_result)
clear_btn.pack(pady=5)
clear_btn.bind("<Enter>", on_enter)
clear_btn.bind("<Leave>", on_leave)

# ----------------- Footer -----------------
footer_label = tk.Label(root, text="© 2025 Yogyata Thapa", bg="white", fg="black", font=("Arial", 10))
footer_label.pack(side="bottom", pady=5)

root.mainloop()
