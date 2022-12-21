from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from turtle import speed
import cv2 as cv 
import numpy as np
import os
import time
import pyttsx3
engine = pyttsx3.init()

class Blank_eyes(App):
    def build(self):
        #returns a window object with all it's widgets
        self.window = GridLayout()
        self.window.cols = 1
        self.window.size_hint = (0.6, 0.7)
        self.window.pos_hint = {"center_x": 0.5, "center_y":0.5}

        # image widget
        self.window.add_widget(Image(source="assets/logo.png"))


        # button widget
        self.button = Button(
                      text= "Start",
                      size_hint= (.5,.5),
                      bold= True,
                      background_color ='white',
                      #remove darker overlay of background colour
                      # background_normal = ""
                      )
        self.button.bind(on_press=self.callback)
        self.window.add_widget(self.button)

        return self.window

    def callback(self, instance):
        # Distance constants 
        KNOWN_DISTANCE = 45 #INCHES
        PERSON_WIDTH = 16 #INCHES
        MOBILE_WIDTH = 3.0 #INCHES
        CAT_WIDTH = 7
        BED_WIDTH = 20
        TABLE_WIDTH = 40
        CAR_WIDTH = 70
        CYCLE_WIDTH = 25
        CHAIR_WIDTH = 18
        SOFA_WIDTH = 90
        TV_WIDTH = 24
        LAPTOP_WIDTH = 18



        # Object detector constant 
        CONFIDENCE_THRESHOLD = 0.4
        NMS_THRESHOLD = 0.3

        # colors for object detected
        COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        GREEN =(0,255,0)
        BLACK =(0,0,0)
        # defining fonts 
        FONTS = cv.FONT_HERSHEY_COMPLEX
        # getting class names from classes.txt file 
        class_names = []
        with open("assets/classes.txt", "r") as f:
            class_names = [cname.strip() for cname in f.readlines()]
        #  setttng up opencv net
        yoloNet = cv.dnn.readNet('assets/yolov4-tiny.weights', 'assets/yolov4-tiny.cfg')


        model = cv.dnn_DetectionModel(yoloNet)
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        # object detector funciton /method
        def object_detector(image):
            classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            # creating empty list to add objects data
            data_list =[]
            for (classid, score, box) in zip(classes, scores, boxes):
                # define color of each, object based on its class id 
                color= COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid], score)
                # draw rectangle on and label on object
                cv.rectangle(image, box, color, 2)
                cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
            
                # getting the data 
                # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
                if classid ==0: # person class id 
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                elif classid ==67: #cell phone
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                elif classid ==15: # cat
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)]) 
                elif classid ==59: # bed
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                elif classid ==60: # table
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                elif classid ==2: # car
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                elif classid ==1: # cycle
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                elif classid ==56: # chair
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
                elif classid ==57: # sofa
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])   
                elif classid ==62: # tv
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])    
                elif classid ==63: # laptop
                    data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])     
                # if you want inclulde more classes then you have to simply add more [elif] statements here
                # returning list containing the object data. 
            return data_list

        def focal_length_finder (measured_distance, real_width, width_in_rf):
            focal_length = (width_in_rf * measured_distance) / real_width
            return focal_length
        # distance finder function 
        def distance_finder(focal_length, real_object_width, width_in_frmae):
            distance = (real_object_width * focal_length) / width_in_frmae
            distance = round(distance)
            return distance

        # reading the reference image from dir 
        ref_person = cv.imread('assets/image14.png')
        ref_mobile = cv.imread('assets/image4.png')
        ref_cat = cv.imread('assets/cat.png')
        ref_bed = cv.imread('assets/bed.jpg')
        ref_car = cv.imread('assets/car.jpg')
        ref_cycle = cv.imread('assets/cycle.jpg')
        ref_chair = cv.imread('assets/chair.jpg')
        ref_table = cv.imread('assets/table.png')
        ref_sofa = cv.imread('assets/sofa.jpg')
        ref_tv = cv.imread('assets/tv.jpg')
        ref_laptop = cv.imread('assets/laptop.jpg')

        #mobile
        mobile_data = object_detector(ref_mobile)
        mobile_width_in_rf = mobile_data[0][1]
        #person
        person_data = object_detector(ref_person)
        person_width_in_rf = person_data[0][1]
        #cat
        cat_data = object_detector(ref_cat)
        cat_width_in_rf = cat_data[0][1]
        #bed
        bed_data = object_detector(ref_bed)
        bed_width_in_rf = bed_data[0][1]
        #dining table
        table_data = object_detector(ref_table)
        table_width_in_rf = table_data[0][1]
        #car
        car_data = object_detector(ref_car)
        car_width_in_rf = car_data[0][1]
        #cycle
        cycle_data = object_detector(ref_cycle)
        cycle_width_in_rf = cycle_data[0][1]
        #chair
        chair_data = object_detector(ref_chair)
        chair_width_in_rf = chair_data[0][1]
        #sofa
        sofa_data = object_detector(ref_sofa)
        sofa_width_in_rf = sofa_data[0][1]
        #tv
        tv_data = object_detector(ref_tv)
        tv_width_in_rf = tv_data[0][1]
        #laptop
        laptop_data = object_detector(ref_laptop)
        laptop_width_in_rf = laptop_data[0][1]

        print("Hello")

        # finding focal length
        #person
        focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
        #mobile
        focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
        #cat
        focal_cat = focal_length_finder(KNOWN_DISTANCE, CAT_WIDTH, cat_width_in_rf)
        #bed
        focal_bed = focal_length_finder(KNOWN_DISTANCE, BED_WIDTH, bed_width_in_rf)
        #dining table
        focal_table = focal_length_finder(KNOWN_DISTANCE, TABLE_WIDTH, table_width_in_rf)
        #car
        focal_car = focal_length_finder(KNOWN_DISTANCE, CAR_WIDTH, car_width_in_rf)
        #cycle
        focal_cycle = focal_length_finder(KNOWN_DISTANCE, CYCLE_WIDTH, cycle_width_in_rf)
        #chair
        focal_chair = focal_length_finder(KNOWN_DISTANCE, CHAIR_WIDTH, chair_width_in_rf)
        #sofa
        focal_sofa = focal_length_finder(KNOWN_DISTANCE, SOFA_WIDTH, sofa_width_in_rf)
        #tv
        focal_tv = focal_length_finder(KNOWN_DISTANCE, TV_WIDTH, tv_width_in_rf)
        #laptop
        focal_laptop = focal_length_finder(KNOWN_DISTANCE, LAPTOP_WIDTH, laptop_width_in_rf)

        cap = cv.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            data = object_detector(frame)
            for d in data:
                #person 
                if d[0] =='person':
                    distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #mobile
                elif d[0] =='cell phone':
                    distance = distance_finder (focal_mobile, MOBILE_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #cat
                elif d[0] =='cat':
                    distance = distance_finder (focal_cat, CAT_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #bed
                elif d[0] =='bed':
                    distance = distance_finder (focal_bed, BED_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #dining table
                elif d[0] =='diningtable':
                    distance = distance_finder (focal_table, TABLE_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #car
                elif d[0] =='car':
                    distance = distance_finder (focal_car, CAR_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #cycle
                elif d[0] =='bicycle':
                    distance = distance_finder (focal_cycle, CYCLE_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #chair
                elif d[0] =='chair':
                    distance = distance_finder (focal_chair, CHAIR_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #sofa
                elif d[0] =='sofa':
                    distance = distance_finder (focal_sofa, SOFA_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #tv
                elif d[0] =='tvmonitor':
                    distance = distance_finder (focal_tv, TV_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                #laptop
                elif d[0] =='laptop':
                    distance = distance_finder (focal_laptop, LAPTOP_WIDTH, d[1])
                    engine.say(d[0])
                    engine.runAndWait()
                    engine.say(str(distance))
                    engine.runAndWait()
                    x, y = d[2]
 
                print(f"Object : {d[0]} || Distance : {round(distance)}cm")  
                    
                cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
                cv.putText(frame, f'Dis: {round(distance,2)} cm', (x+5,y+13), FONTS, 0.48, GREEN, 2)
            cv.imshow('frame',frame)
            key = cv.waitKey(1)
            if key ==ord('q'):
                break
        cv.destroyAllWindows()
        cap.release()


# run Say Hello App Calss
if __name__ == "__main__":
    Blank_eyes().run()