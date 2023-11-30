from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from Task2.preprocessing import PreProcessing


class Task2:
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
        self.num_hidden_layers_value = StringVar(value="")
        self.num_hidden_layers_path = PhotoImage(file="../Neural-Project/Photos/Task2/num_hidden_layers.png")
        self.num_hidden_layers_image_entry = Label(self.root, image=self.num_hidden_layers_path, bg=self.mainColor)
        self.num_hidden_layers_entry = Entry(self.root, width=11, font=("arial", 14), bd=0,
                                             textvariable=self.num_hidden_layers_value,
                                             background=self.mainColor, foreground=self.foregroundColor)

        self.num_neurons_value = StringVar(value="")
        self.num_neurons_path = PhotoImage(file="../Neural-Project/Photos/Task2/num_neurons.png")
        self.num_neurons_image_entry = Label(self.root, image=self.num_neurons_path, bg=self.mainColor)
        self.num_neurons_entry = Entry(self.root, width=11, font=("arial", 14), bd=0,
                                       textvariable=self.num_neurons_value,
                                       background=self.mainColor, foreground=self.foregroundColor)

        self.learning_rate_value = StringVar(value="")
        self.learning_rate_path = PhotoImage(file="../Neural-Project/Photos/Task2/learning_rate.png")
        self.learning_rate_image_entry = Label(self.root, image=self.learning_rate_path, bg=self.mainColor)
        self.learning_rate_entry = Entry(self.root, width=11, font=("arial", 14), bd=0,
                                         textvariable=self.learning_rate_value,
                                         background=self.mainColor, foreground=self.foregroundColor)

        self.epochs_value = StringVar(value="")
        self.epochs_path = PhotoImage(file="../Neural-Project/Photos/Task2/epochs.png")
        self.epochs_image_entry = Label(self.root, image=self.epochs_path, bg=self.mainColor)
        self.epochs_entry = Entry(self.root, width=11, font=("arial", 14), bd=0,
                                  textvariable=self.epochs_value,
                                  background=self.mainColor, foreground=self.foregroundColor)

        self.activation_function = Image.open("../Neural-Project/Photos/Task2/activation_function.png")
        self.activation_function_image = ImageTk.PhotoImage(self.activation_function)
        self.activation_function_label = Label(self.root, image=self.activation_function_image,
                                               background=self.mainColor)



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

        self.activation_function_value = StringVar(value="none")
        self.sigmoid_image_on = PhotoImage(file="../Neural-Project/Photos/Task2/on/sigmoid.png")
        self.sigmoid_image_off = PhotoImage(file="../Neural-Project/Photos/Task2/off/sigmoid.png")
        self.sigmoid_button = Radiobutton(self.root, variable=self.activation_function_value,
                                          value="adaline", background=self.mainColor, image=self.sigmoid_image_off,
                                          selectimage=self.sigmoid_image_on, activebackground=self.mainColor,
                                          indicatoron=False, bd=0, cursor="hand2")

        self.hyperbolic_tangent_image_on = PhotoImage(file="../Neural-Project/Photos/Task2/on/hyperbolic_tangent.png")
        self.hyperbolic_tangent_image_off = PhotoImage(file="../Neural-Project/Photos/Task2/off/hyperbolic_tangent.png")
        self.hyperbolic_tangent_button = Radiobutton(self.root, variable=self.activation_function_value,
                                                     value="perceptron", background=self.mainColor,
                                                     image=self.hyperbolic_tangent_image_off,
                                                     selectimage=self.hyperbolic_tangent_image_on, activebackground=self.mainColor,
                                                     indicatoron=False, bd=0, cursor="hand2")

        self.run_button_image = PhotoImage(file="../Neural-Project/Photos/Task1/run_btn.png")
        self.run_button = Button(self.root, image=self.run_button_image, borderwidth=0, cursor="hand2", bd=0,
                                 background=self.mainColor, activebackground=self.mainColor)

    def placing_widgets(self):
        self.background2_label.place(x=0, y=0)
        self.main_frame2.place(anchor='center', relx=0.5, rely=0.45)

        self.num_hidden_layers_image_entry.place(anchor='center', relx=0.355, y=120)
        self.num_hidden_layers_entry.place(anchor='center', relx=0.532, y=120)

        self.num_neurons_image_entry.place(anchor='center', relx=0.355, y=170)
        self.num_neurons_entry.place(anchor='center', relx=0.532, y=170)

        self.learning_rate_image_entry.place(anchor='center', relx=0.355, y=220)
        self.learning_rate_entry.place(anchor='center', relx=0.532, y=220)

        self.epochs_image_entry.place(anchor='center', relx=0.355, y=270)
        self.epochs_entry.place(anchor='center', relx=0.532, y=270)

        self.activation_function_label.place(anchor='center', relx=0.265, y=340)

        self.sigmoid_button.place(anchor='center', relx=0.17, y=380)
        self.hyperbolic_tangent_button.place(anchor='center', relx=0.43, y=380)

        self.bias_checkbox.place(anchor='center', relx=0.23, y=450)

        self.run_button.place(anchor='center', relx=0.67, y=458)
        self.image_label.pack()

