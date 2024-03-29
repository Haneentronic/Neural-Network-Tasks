from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from Task1.preprocessing import PreProcessing
from Task1.perceptron import Perceptron
from Task1.adaline import Adaline


class Task1:
    def __init__(self):
        self.mainColor = 'white'
        self.foregroundColor = '#FF8888'
        self.secondColor = '#6FAF07'
        self.font = "Comic Sans MS"

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
        self.area_checkbox_value = StringVar(value="None")
        self.area_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/area_on.png")
        self.area_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/area_off.png")
        self.area_checkbox = Checkbutton(self.root, variable=self.area_checkbox_value, background=self.mainColor,
                                         image=self.area_image_off, selectimage=self.area_image_on,
                                         activebackground=self.mainColor,
                                         foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                         onvalue="Area", offvalue="None", command=self.update_features_checkbox)

        self.perimeter_checkbox_value = StringVar(value="None")
        self.perimeter_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/perimeter_on.png")
        self.perimeter_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/perimeter_off.png")
        self.perimeter_checkbox = Checkbutton(self.root, variable=self.perimeter_checkbox_value,
                                              background=self.mainColor,
                                              image=self.perimeter_image_off, selectimage=self.perimeter_image_on,
                                              activebackground=self.mainColor,
                                              foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                              onvalue="Perimeter", offvalue="None",
                                              command=self.update_features_checkbox)

        self.roundnes_checkbox_value = StringVar(value="None")
        self.roundnes_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/roundnes_on.png")
        self.roundnes_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/roundnes_off.png")
        self.roundnes_checkbox = Checkbutton(self.root, variable=self.roundnes_checkbox_value,
                                             background=self.mainColor,
                                             image=self.roundnes_image_off, selectimage=self.roundnes_image_on,
                                             activebackground=self.mainColor,
                                             foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                             onvalue="roundnes", offvalue="None", command=self.update_features_checkbox)

        self.major_checkbox_value = StringVar(value="None")
        self.major_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/major_on.png")
        self.major_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/major_off.png")
        self.major_checkbox = Checkbutton(self.root, variable=self.major_checkbox_value,
                                          background=self.mainColor,
                                          image=self.major_image_off, selectimage=self.major_image_on,
                                          activebackground=self.mainColor,
                                          foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                          onvalue="MajorAxisLength", offvalue="None",
                                          command=self.update_features_checkbox)

        self.minor_checkbox_value = StringVar(value="None")
        self.minor_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/minor_on.png")
        self.minor_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/minor_off.png")
        self.minor_checkbox = Checkbutton(self.root, variable=self.minor_checkbox_value,
                                          background=self.mainColor,
                                          image=self.minor_image_off, selectimage=self.minor_image_on,
                                          activebackground=self.mainColor,
                                          foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                          onvalue="MinorAxisLength", offvalue="None",
                                          command=self.update_features_checkbox)

        self.select_class = Image.open("../Neural-Project/Photos/Task1/select_classes.png")
        self.select_class_image = ImageTk.PhotoImage(self.select_class)
        self.select_class_label = Label(self.root, image=self.select_class_image, background=self.mainColor)

        self.bombay_checkbox_value = StringVar(value="None")
        self.bombay_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/c1_on.png")
        self.bombay_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/c1_off.png")
        self.bombay_checkbox = Checkbutton(self.root, variable=self.bombay_checkbox_value,
                                           background=self.mainColor,
                                           image=self.bombay_image_off, selectimage=self.bombay_image_on,
                                           activebackground=self.mainColor,
                                           foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                           onvalue="BOMBAY", offvalue="None", command=self.update_classes_checkbox)

        self.cali_checkbox_value = StringVar(value="None")
        self.cali_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/c2_on.png")
        self.cali_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/c2_off.png")
        self.cali_checkbox = Checkbutton(self.root, variable=self.cali_checkbox_value,
                                         background=self.mainColor,
                                         image=self.cali_image_off, selectimage=self.cali_image_on,
                                         activebackground=self.mainColor,
                                         foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                         onvalue="CALI", offvalue="None", command=self.update_classes_checkbox)

        self.sira_checkbox_value = StringVar(value="None")
        self.sira_image_on = PhotoImage(file="../Neural-Project/Photos/Task1/on/c3_on.png")
        self.sira_image_off = PhotoImage(file="../Neural-Project/Photos/Task1/off/c3_off.png")
        self.sira_checkbox = Checkbutton(self.root, variable=self.sira_checkbox_value,
                                         background=self.mainColor,
                                         image=self.sira_image_off, selectimage=self.sira_image_on,
                                         activebackground=self.mainColor,
                                         foreground=self.secondColor, bd=0, indicatoron=False, cursor="hand2",
                                         onvalue="SIRA", offvalue="None", command=self.update_classes_checkbox)

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

        self.bias_checkbox_value = IntVar(value=1)
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
                                 background=self.mainColor, activebackground=self.mainColor,
                                 command=lambda: self.run())

        self.feature1 = Image.open("../Neural-Project/Photos/Task1/input2.png")
        self.feature1_image = ImageTk.PhotoImage(self.feature1)
        self.feature1_label = Label(self.root, image=self.feature1_image, background=self.mainColor)

        self.feature1_value = StringVar(value="none")
        self.feature1_entry = Entry(self.root, width=14, font=("arial", 14), bd=0,
                               textvariable=self.feature1_value,
                               background=self.mainColor, foreground=self.foregroundColor)

        self.feature2 = Image.open("../Neural-Project/Photos/Task1/input2.png")
        self.feature2_image = ImageTk.PhotoImage(self.feature2)
        self.feature2_label = Label(self.root, image=self.feature2_image, background=self.mainColor)

        self.feature2_value = StringVar(value="none")
        self.feature2_entry = Entry(self.root, width=14, font=("arial", 14), bd=0,
                               textvariable=self.feature2_value,
                               background=self.mainColor, foreground=self.foregroundColor)

        self.classify_sample_button_image = PhotoImage(file="../Neural-Project/Photos/Task1/classify_sample_btn.png")
        self.classify_sample_button = Button(self.root, image=self.classify_sample_button_image, borderwidth=0, cursor="hand2", bd=0,
                                 background=self.mainColor, activebackground=self.mainColor,
                                 command=lambda: self.classify())

    def run(self):
        features_list = []
        if self.area_checkbox_value.get() != "None":
            features_list.append(self.area_checkbox_value.get())
        if self.perimeter_checkbox_value.get() != "None":
            features_list.append(self.perimeter_checkbox_value.get())
        if self.roundnes_checkbox_value.get() != "None":
            features_list.append(self.roundnes_checkbox_value.get())
        if self.major_checkbox_value.get() != "None":
            features_list.append(self.major_checkbox_value.get())
        if self.minor_checkbox_value.get() != "None":
            features_list.append(self.minor_checkbox_value.get())
        class_list = []
        if self.bombay_checkbox_value.get() != "None":
            class_list.append(self.bombay_checkbox_value.get())
        if self.cali_checkbox_value.get() != "None":
            class_list.append(self.cali_checkbox_value.get())
        if self.sira_checkbox_value.get() != "None":
            class_list.append(self.sira_checkbox_value.get())

        preprocessing = PreProcessing()
        preprocessing.read_data("Task1/Dry_Bean_Dataset.csv", features_list, class_list)
        preprocessing.split_data(40)
        preprocessing.null_handel()
        preprocessing.normalize_train_data()
        preprocessing.normalize_test_data()

        if self.algorithm_value.get() == "perceptron":
            self.o = Perceptron(preprocessing, int(self.epochs_value.get()), float(self.learning_rate_entry.get()),
                           self.bias_checkbox_value.get())
            self.o.perceptron_train()
            self.o.perceptron_test()
            print("Perceptron Accuracy: ", self.o.accuracy_score())
            print("-------------")
            self.o.plot_confusion_matrix(self.o.confusion_matrix(), class_list)
            self.o.plotting()

        elif self.algorithm_value.get() == "adaline":
            self.adaline = Adaline()

            self.adaline.train(preprocessing.x_train, preprocessing.y_train, self.bias_checkbox_value.get()
                          , float(self.learning_rate_entry.get()), float(self.mse_entry.get()))

            self.adaline.plot_decision_boundary(preprocessing.x_train, preprocessing.y_train, self.bias_checkbox_value.get())

            self.adaline.test(preprocessing.x_test, preprocessing.y_test, self.bias_checkbox_value.get())
            print("-------------")

    def classify(self):
        sample = [float(self.feature1_entry.get()), float(self.feature2_entry.get())]
        # print(sample)

        if self.algorithm_value.get() == "adaline":
            pred = self.adaline.predict(x=sample, b=self.bias_checkbox_value.get())

            print("Sample Prediction: ", pred)
            print("-------------")

        elif self.algorithm_value.get() == "perceptron":
            pred = self.o.predict(sample, self.bias_checkbox_value.get())

            print("Sample Prediction: ", pred)
            print("-------------")

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

        self.bombay_checkbox.place(anchor='center', relx=0.14, y=295)
        self.cali_checkbox.place(anchor='center', relx=0.24, y=295)
        self.sira_checkbox.place(anchor='center', relx=0.34, y=295)

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

        self.perceptron_button.place(anchor='center', relx=0.23, y=440)
        self.adaline_button.place(anchor='center', relx=0.45, y=440)

        self.run_button.place(anchor='center', relx=0.67, y=458)

        self.feature1_label.place(anchor='center', relx=0.198, y=500)
        self.feature1_entry.place(anchor='center', relx=0.198, y=500)

        self.feature2_label.place(anchor='center', relx=0.385, y=500)
        self.feature2_entry.place(anchor='center', relx=0.385, y=500)

        self.classify_sample_button.place(anchor='center', relx=0.54, y=500)

        self.image_label.pack()

    def update_classes_checkbox(self):
        i = 0
        if self.bombay_checkbox_value.get() == "BOMBAY":
            i = i + 1
        if self.cali_checkbox_value.get() == "CALI":
            i = i + 1
        if self.sira_checkbox_value.get() == "SIRA":
            i = i + 1

        if i >= 2:
            if self.bombay_checkbox_value.get() != "BOMBAY":
                self.bombay_checkbox.config(state="disabled")
            if self.cali_checkbox_value.get() != "CALI":
                self.cali_checkbox.config(state="disabled")
            if self.sira_checkbox_value.get() != "SIRA":
                self.sira_checkbox.config(state="disabled")

        else:
            self.bombay_checkbox.config(state="normal")
            self.cali_checkbox.config(state="normal")
            self.sira_checkbox.config(state="normal")

    def update_features_checkbox(self):
        i = 0
        if self.area_checkbox_value.get() == "Area":
            i = i + 1
        if self.perimeter_checkbox_value.get() == "Perimeter":
            i = i + 1
        if self.roundnes_checkbox_value.get() == "roundnes":
            i = i + 1
        if self.major_checkbox_value.get() == "MajorAxisLength":
            i = i + 1
        if self.minor_checkbox_value.get() == "MinorAxisLength":
            i = i + 1
        if i >= 2:
            if self.area_checkbox_value.get() != "Area":
                self.area_checkbox.config(state="disabled")
            if self.perimeter_checkbox_value.get() != "Perimeter":
                self.perimeter_checkbox.config(state="disabled")
            if self.roundnes_checkbox_value.get() != "roundnes":
                self.roundnes_checkbox.config(state="disabled")
            if self.major_checkbox_value.get() != "MajorAxisLength":
                self.major_checkbox.config(state="disabled")
            if self.minor_checkbox_value.get() != "MinorAxisLength":
                self.minor_checkbox.config(state="disabled")
        else:
            self.area_checkbox.config(state="normal")
            self.perimeter_checkbox.config(state="normal")
            self.roundnes_checkbox.config(state="normal")
            self.major_checkbox.config(state="normal")
            self.minor_checkbox.config(state="normal")
