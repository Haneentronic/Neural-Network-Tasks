from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk


class Task1:
    def __init__(self):
        self.mainColor = 'white'
        self.foregroundColor = '#FF8888'
        self.secondColor = '#6FAF07'

        self.root = tk.Tk()
        self.root.geometry("1000x563")

        # setting background image
        self.setting_background()

        # Creating widgets
        self.create_widgets()

        # Placing widgets on the screen
        self.placing_widgets()

        self.root.mainloop()

    def setting_background(self):
        self.image = Image.open("../Neural-Project/Photos/background_sub.png")
        self.background2_image = ImageTk.PhotoImage(self.image)
        self.background2_label = Label(self.root, image=self.background2_image)
        self.main_frame2 = tk.Frame(self.root, borderwidth=0, background=self.mainColor)
        self.image_label = tk.Label(self.main_frame2, image="", borderwidth=0, background=self.mainColor)

    def create_widgets(self):
        self.select_feature = Image.open("../Neural-Project/Photos/Task1/select_feature.png")
        self.select_feature_image = ImageTk.PhotoImage(self.select_feature)
        self.select_feature_label = Label(self.root, image=self.select_feature_image, background=self.mainColor)

        # select features
        self.area_checkbox_value = IntVar(value=-1)
        self.area_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/area_on.png")
        self.area_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/area_off.png")
        self.area_checkbox = Checkbutton(self.root, variable=self.area_checkbox_value, background=self.mainColor,
                                         image=self.area_image_off, selectimage=self.area_image_on,
                                         activebackground=self.mainColor,
                                         foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                         onvalue=1, offvalue=0, command=self.update_features_checkbox)

        self.perimeter_checkbox_value = IntVar(value=-1)
        self.perimeter_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/perimeter_on.png")
        self.perimeter_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/perimeter_off.png")
        self.perimeter_checkbox = Checkbutton(self.root, variable=self.perimeter_checkbox_value,
                                              background=self.mainColor,
                                              image=self.perimeter_image_off, selectimage=self.perimeter_image_on,
                                              activebackground=self.mainColor,
                                              foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                              onvalue=1, offvalue=0, command=self.update_features_checkbox)

        self.roundnes_checkbox_value = IntVar(value=-1)
        self.roundnes_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/roundnes_on.png")
        self.roundnes_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/roundnes_off.png")
        self.roundnes_checkbox = Checkbutton(self.root, variable=self.roundnes_checkbox_value,
                                             background=self.mainColor,
                                             image=self.roundnes_image_off, selectimage=self.roundnes_image_on,
                                             activebackground=self.mainColor,
                                             foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                             onvalue=1, offvalue=0, command=self.update_features_checkbox)

        self.major_checkbox_value = IntVar(value=-1)
        self.major_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/major_on.png")
        self.major_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/major_off.png")
        self.major_checkbox = Checkbutton(self.root, variable=self.major_checkbox_value,
                                          background=self.mainColor,
                                          image=self.major_image_off, selectimage=self.major_image_on,
                                          activebackground=self.mainColor,
                                          foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                          onvalue=1, offvalue=0, command=self.update_features_checkbox)

        self.minor_checkbox_value = IntVar(value=-1)
        self.minor_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/minor_on.png")
        self.minor_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/minor_off.png")
        self.minor_checkbox = Checkbutton(self.root, variable=self.minor_checkbox_value,
                                          background=self.mainColor,
                                          image=self.minor_image_off, selectimage=self.minor_image_on,
                                          activebackground=self.mainColor,
                                          foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                          onvalue=1, offvalue=0, command=self.update_features_checkbox)

        self.select_class = Image.open("../Neural-Project/Photos/Task1/select_classes.png")
        self.select_class_image = ImageTk.PhotoImage(self.select_class)
        self.select_class_label = Label(self.root, image=self.select_class_image, background=self.mainColor)

        self.c1_checkbox_value = IntVar(value=-1)
        self.c1_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/c1_on.png")
        self.c1_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/c1_off.png")
        self.c1_checkbox = Checkbutton(self.root, variable=self.c1_checkbox_value,
                                       background=self.mainColor,
                                       image=self.c1_image_off, selectimage=self.c1_image_on,
                                       activebackground=self.mainColor,
                                       foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2", onvalue=1,
                                       offvalue=0, command=self.update_classes_checkbox)

        self.c2_checkbox_value = IntVar(value=-1)
        self.c2_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/c2_on.png")
        self.c2_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/c2_off.png")
        self.c2_checkbox = Checkbutton(self.root, variable=self.c2_checkbox_value,
                                       background=self.mainColor,
                                       image=self.c2_image_off, selectimage=self.c2_image_on,
                                       activebackground=self.mainColor,
                                       foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2", onvalue=1,
                                       offvalue=0, command=self.update_classes_checkbox)

        self.c3_checkbox_value = IntVar(value=-1)
        self.c3_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/c3_on.png")
        self.c3_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/c3_off.png")
        self.c3_checkbox = Checkbutton(self.root, variable=self.c3_checkbox_value,
                                       background=self.mainColor,
                                       image=self.c3_image_off, selectimage=self.c3_image_on,
                                       activebackground=self.mainColor,
                                       foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2", onvalue=1,
                                       offvalue=0, command=self.update_classes_checkbox)

        self.learning_rate = Image.open("../Neural-Project/Photos/Task1/learning_rate.png")
        self.learning_rate_image = ImageTk.PhotoImage(self.learning_rate)
        self.learning_rate_label = Label(self.root, image=self.learning_rate_image, background=self.mainColor)

        self.learning_rate_value = StringVar(value="")
        self.learning_rate_path = PhotoImage(file="../Neural-Project/Photos/Task1/input.png")
        self.learning_rate_image_entry = Label(self.root, image=self.learning_rate_path, bg=self.mainColor)
        self.learning_rate_entry = Entry(self.root, width=9, font=("arial", 14), bd=0,
                                         textvariable=self.learning_rate_value,
                                         background=self.mainColor, foreground=self.foregroundColor)

        self.epochs = Image.open("../Neural-Project/Photos/Task1/epochs.png")
        self.epochs_image = ImageTk.PhotoImage(self.epochs)
        self.epochs_label = Label(self.root, image=self.epochs_image, background=self.mainColor)

        self.epochs_value = StringVar(value="")
        self.epochs_path = PhotoImage(file="../Neural-Project/Photos/Task1/input.png")
        self.epochs_image_entry = Label(self.root, image=self.epochs_path, bg=self.mainColor)
        self.epochs_entry = Entry(self.root, width=9, font=("arial", 14), bd=0,
                                  textvariable=self.epochs_value,
                                  background=self.mainColor, foreground=self.foregroundColor)

        self.mse = Image.open("../Neural-Project/Photos/Task1/mse.png")
        self.mse_image = ImageTk.PhotoImage(self.mse)
        self.mse_label = Label(self.root, image=self.mse_image, background=self.mainColor)

        self.mse_value = StringVar(value="")
        self.mse_path = PhotoImage(file="../Neural-Project/Photos/Task1/input.png")
        self.mse_image_entry = Label(self.root, image=self.mse_path, bg=self.mainColor)
        self.mse_entry = Entry(self.root, width=9, font=("arial", 14), bd=0,
                               textvariable=self.mse_value,
                               background=self.mainColor, foreground=self.foregroundColor)

        self.bias_checkbox_value = IntVar(value=-1)
        self.bias_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/bias.png")
        self.bias_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/not_bias.png")
        self.bias_checkbox = Checkbutton(self.root, variable=self.bias_checkbox_value,
                                         background=self.mainColor,
                                         image=self.bias_image_off, selectimage=self.bias_image_on,
                                         activebackground=self.mainColor,
                                         foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                         onvalue=1,
                                         offvalue=0)

        self.algorithm_value = StringVar(value="none")
        self.perceptron_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/perceptron_on.png")
        self.perceptron_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/perceptron_off.png")
        self.perceptron_button = Radiobutton(self.root, variable=self.algorithm_value,
                                             value="perceptron", background=self.mainColor,
                                             image=self.perceptron_image_off,
                                             selectimage=self.perceptron_image_on, activebackground=self.mainColor,
                                             indicatoron=False, bd=0, cursor="hand2")

        self.adaline_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/adaline_on.png")
        self.adaline_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/adaline_off.png")
        self.adaline_button = Radiobutton(self.root, variable=self.algorithm_value,
                                          value="adaline", background=self.mainColor, image=self.adaline_image_off,
                                          selectimage=self.adaline_image_on, activebackground=self.mainColor,
                                          indicatoron=False, bd=0, cursor="hand2")

        self.run_button_image = PhotoImage(file="../Neural-Project/Photos/Task1/run_btn.png")
        self.run_button = Button(self.root, image=self.run_button_image, borderwidth=0, cursor="hand2", bd=0,
                                 background=self.mainColor, activebackground=self.mainColor)

    def placing_widgets(self):
        self.background2_label.place(x=0, y=0)
        self.main_frame2.place(anchor='center', relx=0.5, rely=0.45)

        self.select_feature_label.place(anchor='center', relx=0.31, y=95)

        self.area_checkbox.place(anchor='center', relx=0.16, y=130)
        self.perimeter_checkbox.place(anchor='center', relx=0.32, y=130)
        self.roundnes_checkbox.place(anchor='center', relx=0.52, y=130)
        self.major_checkbox.place(anchor='center', relx=0.26, y=165)
        self.minor_checkbox.place(anchor='center', relx=0.26, y=200)

        self.select_class_label.place(anchor='center', relx=0.3, y=260)

        self.c1_checkbox.place(anchor='center', relx=0.14, y=295)
        self.c2_checkbox.place(anchor='center', relx=0.24, y=295)
        self.c3_checkbox.place(anchor='center', relx=0.34, y=295)

        self.learning_rate_label.place(anchor='center', relx=0.22, y=350)
        self.learning_rate_image_entry.place(anchor='center', relx=0.415, y=350)
        self.learning_rate_entry.place(anchor='center', relx=0.415, y=350)

        self.epochs_label.place(anchor='center', relx=0.56, y=350)
        self.epochs_image_entry.place(anchor='center', relx=0.69, y=350)
        self.epochs_entry.place(anchor='center', relx=0.69, y=350)

        self.mse_label.place(anchor='center', relx=0.225, y=390)
        self.mse_image_entry.place(anchor='center', relx=0.415, y=395)
        self.mse_entry.place(anchor='center', relx=0.415, y=395)

        self.bias_checkbox.place(anchor='center', relx=0.626, y=395)

        self.perceptron_button.place(anchor='center', relx=0.23, y=460)
        self.adaline_button.place(anchor='center', relx=0.45, y=460)

        self.run_button.place(anchor='center', relx=0.67, y=458)
        self.image_label.pack()

    def update_classes_checkbox(self):
        i = 0
        if self.c1_checkbox_value.get() == 1:
            i = i + 1
        if self.c2_checkbox_value.get() == 1:
            i = i + 1
        if self.c3_checkbox_value.get() == 1:
            i = i + 1

        if i >= 2:
            if self.c1_checkbox_value.get() != 1:
                self.c1_checkbox.config(state="disabled")
            if self.c2_checkbox_value.get() != 1:
                self.c2_checkbox.config(state="disabled")
            if self.c3_checkbox_value.get() != 1:
                self.c3_checkbox.config(state="disabled")

        else:
            self.c1_checkbox.config(state="normal")
            self.c2_checkbox.config(state="normal")
            self.c3_checkbox.config(state="normal")

    def update_features_checkbox(self):
        i = 0
        if self.area_checkbox_value.get() == 1:
            i = i + 1
        if self.perimeter_checkbox_value.get() == 1:
            i = i + 1
        if self.roundnes_checkbox_value.get() == 1:
            i = i + 1
        if self.major_checkbox_value.get() == 1:
            i = i + 1
        if self.minor_checkbox_value.get() == 1:
            i = i + 1
        if i >= 2:
            if self.area_checkbox_value.get() != 1:
                self.area_checkbox.config(state="disabled")
            if self.perimeter_checkbox_value.get() != 1:
                self.perimeter_checkbox.config(state="disabled")
            if self.roundnes_checkbox_value.get() != 1:
                self.roundnes_checkbox.config(state="disabled")
            if self.major_checkbox_value.get() != 1:
                self.major_checkbox.config(state="disabled")
            if self.minor_checkbox_value.get() != 1:
                self.minor_checkbox.config(state="disabled")
        else:
            self.area_checkbox.config(state="normal")
            self.perimeter_checkbox.config(state="normal")
            self.roundnes_checkbox.config(state="normal")
            self.major_checkbox.config(state="normal")
            self.minor_checkbox.config(state="normal")
