import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

import scipy.io

# Load the MNIST data
mnist = scipy.io.loadmat('mnist-original.mat')
data = mnist['data'].T
labels = mnist['label'][0]

# Function to display the selected digit
def display_digit(index):
    digit = data[index].reshape(28, 28)
    img = Image.fromarray(digit * 255).convert('L')
    img = img.resize((280, 280), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

# Create the main window
root = tk.Tk()
root.title("MNIST Digit Classifier")

# Create a dropdown menu to select the digit
label = tk.Label(root, text="Choose a digit index (0-9):")
label.pack(pady=10)

digit_var = tk.IntVar()
digit_menu = ttk.Combobox(root, textvariable=digit_var)
digit_menu['values'] = list(range(10))
digit_menu.current(0)
digit_menu.pack(pady=10)

# Create a panel to display the digit
panel = tk.Label(root)
panel.pack(pady=10)

# Display the initial digit
display_digit(0)

# Update the displayed digit when a new index is selected
digit_menu.bind("<<ComboboxSelected>>", lambda event: display_digit(digit_var.get()))

# Run the application
root.mainloop()