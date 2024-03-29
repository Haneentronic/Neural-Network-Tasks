from tkinter import *
import tkinter as tk
import pandas as pd
from PIL import Image, ImageTk
from Task2.preprocessing import PreProcessing
from Task2.evaluate_old import Evaluate
from Task2.multi_layer_perceptron import MultiLayerPerceptron, extract_input_and_output


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



        self.bias_checkbox_value = IntVar(value=0)
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
                                          value="sigmoid", background=self.mainColor, image=self.sigmoid_image_off,
                                          selectimage=self.sigmoid_image_on, activebackground=self.mainColor,
                                          indicatoron=False, bd=0, cursor="hand2")

        self.hyperbolic_tangent_image_on = PhotoImage(file="../Neural-Project/Photos/Task2/on/hyperbolic_tangent.png")
        self.hyperbolic_tangent_image_off = PhotoImage(file="../Neural-Project/Photos/Task2/off/hyperbolic_tangent.png")
        self.hyperbolic_tangent_button = Radiobutton(self.root, variable=self.activation_function_value,
                                                     value="tanh", background=self.mainColor,
                                                     image=self.hyperbolic_tangent_image_off,
                                                     selectimage=self.hyperbolic_tangent_image_on, activebackground=self.mainColor,
                                                     indicatoron=False, bd=0, cursor="hand2")

# --------------------------------------------------------------------------------------------------------
        self.user_inputs_path = PhotoImage(file="../Neural-Project/Photos/Task2/user_input.png")
        self.user_inputs_image_entry = Label(self.root, image=self.user_inputs_path, bg=self.mainColor)

        self.area_value = StringVar(value="")
        self.area_entry = Entry(self.root, width=7, font=("arial", 10), bd=0,
                                textvariable=self.area_value,
                                background=self.mainColor, foreground=self.foregroundColor)

        self.perimeter_value = StringVar(value="")
        self.perimeter_entry = Entry(self.root, width=7, font=("arial", 10), bd=0,
                                     textvariable=self.perimeter_value,
                                     background=self.mainColor, foreground=self.foregroundColor)

        self.major_axis_length_value = StringVar(value="")
        self.major_axis_length_entry = Entry(self.root, width=7, font=("arial", 10), bd=0,
                                             textvariable=self.major_axis_length_value,
                                             background=self.mainColor, foreground=self.foregroundColor)

        self.minor_axis_length_value = StringVar(value="")
        self.minor_axis_length_entry = Entry(self.root, width=7, font=("arial", 10), bd=0,
                                             textvariable=self.minor_axis_length_value,
                                             background=self.mainColor, foreground=self.foregroundColor)

        self.roundnes_value = StringVar(value="")
        self.roundnes_entry = Entry(self.root, width=7, font=("arial", 10), bd=0,
                                    textvariable=self.roundnes_value,
                                    background=self.mainColor, foreground=self.foregroundColor)
# ---------------------------------------------------------------------------------------------------------

        self.run_button_image = PhotoImage(file="../Neural-Project/Photos/Task2/classifyBtn.png")
        self.run_button = Button(self.root, image=self.run_button_image, borderwidth=0, cursor="hand2", bd=0,
                                 background=self.mainColor, activebackground=self.mainColor,
                                 command=lambda: self.classify_sample())

        self.train_button_image = PhotoImage(file="../Neural-Project/Photos/Task2/trainBtn.png")
        self.train_button = Button(self.root, image=self.train_button_image, borderwidth=0, cursor="hand2", bd=0,
                                 background=self.mainColor, activebackground=self.mainColor, command=lambda: self.run())

    def placing_widgets(self):
        self.background2_label.place(x=0, y=0)
        self.main_frame2.place(anchor='center', relx=0.5, rely=0.45)

        self.num_hidden_layers_image_entry.place(anchor='center', relx=0.355, y=100)
        self.num_hidden_layers_entry.place(anchor='center', relx=0.532, y=100)

        self.num_neurons_image_entry.place(anchor='center', relx=0.355, y=150)
        self.num_neurons_entry.place(anchor='center', relx=0.532, y=150)

        self.learning_rate_image_entry.place(anchor='center', relx=0.355, y=200)
        self.learning_rate_entry.place(anchor='center', relx=0.532, y=200)

        self.epochs_image_entry.place(anchor='center', relx=0.355, y=250)
        self.epochs_entry.place(anchor='center', relx=0.532, y=250)

        self.activation_function_label.place(anchor='center', relx=0.265, y=300)

        self.sigmoid_button.place(anchor='center', relx=0.17, y=340)
        self.hyperbolic_tangent_button.place(anchor='center', relx=0.43, y=340)

        self.user_inputs_image_entry.place(anchor='center', relx=0.433, y=390)
        self.area_entry.place(anchor='center', relx=0.14, y=402)
        self.perimeter_entry.place(anchor='center', relx=0.2579, y=402)
        self.major_axis_length_entry.place(anchor='center', relx=0.4079, y=402)
        self.minor_axis_length_entry.place(anchor='center', relx=0.572, y=402)
        self.roundnes_entry.place(anchor='center', relx=0.7159, y=402)

        self.bias_checkbox.place(anchor='center', relx=0.23, y=470)

        self.train_button.place(anchor='center', relx=0.55, y=458)
        self.run_button.place(anchor='center', relx=0.68, y=458)
        self.image_label.pack()

    def run(self):
        self.preprocessing = PreProcessing()
        self.preprocessing.read_data("Task2/Dry_Bean_Dataset.csv",
                                ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'roundnes'],
                                ['CALI', 'BOMBAY', 'SIRA'])
        self.preprocessing.handel_all_outliers()
        self.preprocessing.split_data(40)
        self.preprocessing.null_handel()
        self.preprocessing.normalize_train_data()
        self.preprocessing.normalize_test_data()

        hidden_neurons_list = list(self.num_neurons_value.get().split(","))
        hidden_neurons_list = list(map(int, hidden_neurons_list))

        self.mlp = MultiLayerPerceptron(5, hidden_neurons_list, 3, self.activation_function_value.get(), int(self.bias_checkbox_value.get()))
        train_input, train_expected_output = extract_input_and_output(self.preprocessing.x_train, self.preprocessing.y_train)
        self.mlp.train(train_input, train_expected_output,
                  int(self.epochs_value.get()), float(self.learning_rate_value.get()), int(self.bias_checkbox_value.get()))
        train_prediction = self.mlp.predict(self.preprocessing.x_train)
        train_evaluator = Evaluate(train_prediction, train_expected_output, self.mlp.num_outputs)
        train_evaluator.calculate_confusion_matrix()
        print("Train Confusion Matrix: ")
        print(train_evaluator.confusion_matrix)
        print("Train Accuracy: ", train_evaluator.calculate_accuracy())

        test_input, test_expected_output = extract_input_and_output(self.preprocessing.x_test, self.preprocessing.y_test)
        test_prediction = self.mlp.predict(self.preprocessing.x_test)
        test_evaluator = Evaluate(test_prediction, test_expected_output, self.mlp.num_outputs)
        test_evaluator.calculate_confusion_matrix()
        print("Test Confusion Matrix: ")
        print(test_evaluator.confusion_matrix)
        print("Test Accuracy: ", test_evaluator.calculate_accuracy())

    def classify_sample(self):
        data = {'Area': [float(self.area_value.get())],
                'Perimeter': [float(self.perimeter_value.get())],
                'MajorAxisLength': [float(self.major_axis_length_value.get())],
                'MinorAxisLength': [float(self.minor_axis_length_value.get())],
                'roundnes': [float(self.roundnes_value.get())]}
        sample = pd.DataFrame(data)
        sample = self.preprocessing.normalize_sample(sample)
        sample_prediction = self.mlp.predict(sample)
        print("Sample Prediction: ", sample_prediction)

        predicted_class = None
        if sample_prediction[0][0] == 1:
            predicted_class = "BOMBAY"
        elif sample_prediction[0][1] == 1:
            predicted_class = "CALI"
        elif sample_prediction[0][2] == 1:
            predicted_class = "SIRA"
        print("predicted class for ur Inputs is: ", predicted_class)

