from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from Task1.main import task1

def open_task(root, page):
    root.destroy()
    page()


class MainScreen:
    def __init__(self):
        mainColor = 'white'

        root = tk.Tk()
        root.geometry("1000x563")

        # setting background image
        image = Image.open("Photos/background_main.png")
        background_image = ImageTk.PhotoImage(image)
        background_label = Label(root, image=background_image)

        main_frame = tk.Frame(root, borderwidth=0, background=mainColor)
        image_label = tk.Label(main_frame, image="", borderwidth=0, background=mainColor)

        # Creating widgets
        choose_task = Image.open("Photos/choose_task.png")
        choose_task_image = ImageTk.PhotoImage(choose_task)
        choose_task_label = Label(root, image=choose_task_image, background=mainColor)

        task1_button_image = PhotoImage(file="Photos/task1_btn.png")
        task1_button = Button(root, image=task1_button_image, borderwidth=0, cursor="hand2", bd=0,
                              background=mainColor, activebackground=mainColor, command=lambda: open_task(root, task1))


        # Placing widgets on the screen
        background_label.place(x=0, y=0)
        main_frame.place(anchor='center', relx=0.5, rely=0.45)
        choose_task_label.place(anchor='center', relx=0.5, y=200)
        task1_button.place(anchor='center', relx=0.5, y=300)
        image_label.pack()

        root.mainloop()


MainScreen()
