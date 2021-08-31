#!/usr/bin/env python
# coding: utf-8

# In[7]:


import csv
import tkinter
import sklearn

from tkinter import messagebox
from tkinter import *

root = tkinter.Tk()
root.title("AI Diagnostic")
root.geometry("1920x820")
root.configure(bg="white")

# global variables
#stg_v= StringVar()
#med_v = StringVar()
gender_v = StringVar()
age_v = StringVar()
#fhistory_v = StringVar()
#patient_v = StringVar()
#bpcontrol_v = StringVar()
headache_v = StringVar()
breath_v = StringVar()
visual_v = StringVar()
nosebleeding_v = StringVar()
diagnosed_v = StringVar()
systolic_v = StringVar()
diastolic_v = StringVar()
#diet_v = StringVar()

# Set default unchecked
def unchecked():
    gender_v.set(' ')
    age_v.set(' ')
 #   fhistory_v.set(' ')
 #   patient_v.set(' ')
 #   bpcontrol_v.set(' ')
    headache_v.set(' ')
    breath_v.set(' ')
    visual_v.set(' ')
    nosebleeding_v.set(' ')
    diagnosed_v.set(' ')
    systolic_v.set(' ')
    diastolic_v.set(' ')
 #   diet_v.set(' ')
#`   med_v.set(' ')
   # stg_v.set(' ')
unchecked()

#Fonts section
helv26 = "Helvetica 30 bold"
times = "Times 18 bold"
label_text = "Arial 15"
radiobutton_label = "Arial 15"

# Main headings section

main_heading2 = Label(text="AI Diagnostic",bg="white",fg="Orange",font=helv26)
main_heading2.place(x=50, y=20)

# Radio buttons
gender_radiobutton1 = Radiobutton(text="male",bg="white",fg="black",font=radiobutton_label,variable=gender_v,value="1")
gender_radiobutton1.place(x=50, y=530)

gender_radiobutton2 = Radiobutton(text="female",bg="white",fg="black",font=radiobutton_label,variable=gender_v,value="0")
gender_radiobutton2.place(x=50, y=560)

# gender_radiobutton3 = Radiobutton(text="other",bg="white",fg="black",font=radiobutton_label,variable=gender_v,value="other")
# gender_radiobutton3.place(x=50, y=590)

age_radiobutton1 = Radiobutton(text="18-34",bg="white",fg="black",font=radiobutton_label,variable=age_v,value="0")
age_radiobutton1.place(x=50, y=660)

age_radiobutton2 = Radiobutton(text="35-50",bg="white",fg="black",font=radiobutton_label,variable=age_v,value="1")
age_radiobutton2.place(x=50, y=690)

age_radiobutton3 = Radiobutton(text="51-64",bg="white",fg="black",font=radiobutton_label,variable=age_v,value="2")
age_radiobutton3.place(x=50, y=720)

age_radiobutton4 = Radiobutton(text="65+",bg="white",fg="black",font=radiobutton_label,variable=age_v,value="3")
age_radiobutton4.place(x=50, y=750)

# fhistory_radiobutton1 = Radiobutton(text="yes",bg="white",fg="black",font=radiobutton_label,variable=fhistory_v,value="yes")
# fhistory_radiobutton1.place(x=50, y=330)

# fhistory_radiobutton2 = Radiobutton(text="no",bg="white",fg="black",font=radiobutton_label,variable=fhistory_v,value="no")
# fhistory_radiobutton2.place(x=120, y=330)

# patient_radiobutton1 = Radiobutton(text="yes",bg="white",fg="black",font=radiobutton_label,variable=patient_v,value="yes")
# patient_radiobutton1.place(x=50, y=420)

# patient_radiobutton2 = Radiobutton(text="no",bg="white",fg="black",font=radiobutton_label,variable=patient_v,value="no")
# patient_radiobutton2.place(x=120, y=420)

# bpcontrol_radiobutton1 = Radiobutton(text="yes",bg="white",fg="black",font=radiobutton_label,variable=bpcontrol_v,value="yes")
# bpcontrol_radiobutton1.place(x=50, y=510)

# bpcontrol_radiobutton2 = Radiobutton(text="no",bg="white",fg="black",font=radiobutton_label,variable=bpcontrol_v,value="no")
# bpcontrol_radiobutton2.place(x=120, y=510)

# bpcontrol_radiobutton3 = Radiobutton(text="partial",bg="white",fg="black",font=radiobutton_label,variable=bpcontrol_v,value="partial")
# bpcontrol_radiobutton3.place(x=180, y=510)

headache_radiobutton1 = Radiobutton(text="Mild",bg="white",fg="black",font=radiobutton_label,variable=headache_v,value="0")
headache_radiobutton1.place(x=400, y=90)

headache_radiobutton2 = Radiobutton(text="Moderate",bg="white",fg="black",font=radiobutton_label,variable=headache_v,value="1")
headache_radiobutton2.place(x=470, y=90)

headache_radiobutton3 = Radiobutton(text="Sever",bg="white",fg="black",font=radiobutton_label,variable=headache_v,value="2")
headache_radiobutton3.place(x=590, y=90)

breath_radiobutton1 = Radiobutton(text="yes",bg="white",fg="black",font=radiobutton_label,variable=breath_v,value="1")
breath_radiobutton1.place(x=400, y=180)

breath_radiobutton2 = Radiobutton(text="no",bg="white",fg="black",font=radiobutton_label,variable=breath_v,value="0")
breath_radiobutton2.place(x=460, y=180)

visual_radiobutton1 = Radiobutton(text="yes",bg="white",fg="black",font=radiobutton_label,variable=visual_v,value="1")
visual_radiobutton1.place(x=400, y=270)

visual_radiobutton2 = Radiobutton(text="no",bg="white",fg="black",font=radiobutton_label,variable=visual_v,value="0")
visual_radiobutton2.place(x=460, y=270)

nose_radiobutton1 = Radiobutton(text="yes",bg="white",fg="black",font=radiobutton_label,variable=nosebleeding_v,value="1")
nose_radiobutton1.place(x=400, y=360)

nose_radiobutton2 = Radiobutton(text="no",bg="white",fg="black",font=radiobutton_label,variable=nosebleeding_v,value="0")
nose_radiobutton2.place(x=460, y=360)

diagnosed_radiobutton1 = Radiobutton(text="<1 Year",bg="white",fg="black",font=radiobutton_label,variable=diagnosed_v,value="1")
diagnosed_radiobutton1.place(x=400, y=450)

diagnosed_radiobutton2 = Radiobutton(text="1-5 Years",bg="white",fg="black",font=radiobutton_label,variable=diagnosed_v,value="0")
diagnosed_radiobutton2.place(x=520, y=450)

diagnosed_radiobutton3 = Radiobutton(text=">5 Years",bg="white",fg="black",font=radiobutton_label,variable=diagnosed_v,value="2")
diagnosed_radiobutton3.place(x=650, y=450)


bprangeS_radiobutton1 = Radiobutton(text="111-129",bg="white",fg="black",font=radiobutton_label,variable=systolic_v,value="0")
bprangeS_radiobutton1.place(x=400, y=540)

bprangeS_radiobutton2 = Radiobutton(text="121-130",bg="white",fg="black",font=radiobutton_label,variable=systolic_v,value="1")
bprangeS_radiobutton2.place(x=520, y=540)

bprangeS_radiobutton3 = Radiobutton(text="130+",bg="white",fg="black",font=radiobutton_label,variable=systolic_v,value="2")
bprangeS_radiobutton3.place(x=640, y=540)


bpranged_radiobutton1 = Radiobutton(text="81-90",bg="white",fg="black",font=radiobutton_label,variable=diastolic_v,value="2")
bpranged_radiobutton1.place(x=400, y=630)

bpranged_radiobutton2 = Radiobutton(text="91-100",bg="white",fg="black",font=radiobutton_label,variable=diastolic_v,value="3")
bpranged_radiobutton2.place(x=500, y=630)

bpranged_radiobutton3 = Radiobutton(text="100+",bg="white",fg="black",font=radiobutton_label,variable=diastolic_v,value="0")
bpranged_radiobutton3.place(x=600, y=630)

# diet_radiobutton1 = Radiobutton(text="yes",bg="white",fg="black",font=radiobutton_label,variable=diet_v,value="yes")
# diet_radiobutton1.place(x=820, y=100)

# diet_radiobutton2 = Radiobutton(text="no",bg="white",fg="black",font=radiobutton_label,variable=diet_v,value="no")
# diet_radiobutton2.place(x=880, y=100)

# diet_radiobutton3 = Radiobutton(text="partial",bg="white",fg="black",font=radiobutton_label,variable=diet_v,value="partial")
# diet_radiobutton3.place(x=940, y=100)

# Labels
l_headings = Label(text="AI Based Blood Pressure Software is your consultant which can help you how to treat with stage 1 and stage 2 Blood pressure problems. Fill that form then AI Based Software analyze the stage and recommend a medicine according to the patient conditions. ",wraplength=250,justify=LEFT,font=label_text,bg="white")
l_headings.place(x=50, y=100)


l_gender = Label(text="Gender*",font=label_text)
l_gender.place(x=50, y=500)

l_age = Label(text="Age*",font=label_text)
l_age.place(x=50, y=630)

# l_fhistory = Label(text="Family history of Hypertension",font=label_text)
# l_fhistory.place(x=50, y=300)

# l_patient = Label(text="Are you a patient of Hypertension?",font=label_text)
# l_patient.place(x=50, y=380)

# l_controlbp = Label(text="Did the medication you take effectively\n control your blood pressure?",font=label_text)
# l_controlbp.place(x=50, y=460)

l_headach = Label(text="What is the severity of your headache during hypertension?",font=label_text)
l_headach.place(x=400, y=50)

l_breath = Label(text="Do you feel shortness of breath?",font=label_text)
l_breath.place(x=400, y=140)

l_visual = Label(text="Do you feel visual changes?",font=label_text)
l_visual.place(x=400, y=230)

l_nosebleeding = Label(text="Have you ever experienced nose bleeding during hypertension?",font=label_text)
l_nosebleeding.place(x=400, y=320)

l_diagnosed = Label(text="Since when you have been diagnosed as high blood pressure patient?",font=label_text)
l_diagnosed.place(x=400, y=410)

l_systolic = Label(text="What was your recent BP range for last 2 months? (Systolic)",font=label_text)
l_systolic.place(x=400, y=500)

l_diastolic = Label(text="What was your recent BP range for last 2 months? (Diastolic)",font=label_text)
l_diastolic.place(x=400, y=590)

# l_diet = Label(text="Do you take controlled diet and avoid excess\nsalt and spice in your food?",font=label_text)
# l_diet.place(x=820, y=50)
l_rec = Label(text="Do you want a recommended Medication?",font=label_text)
l_rec.place(x=1100, y=150)


#Stage
l_stg = Label(text="Stage",font=label_text)
l_stg.place(x=1100, y=55)

# entry

##stg_entry = Entry(root, text ="",textvar=stg_v,bg="light steel blue", width=36 )
# med_entry.insert(0,"Panadol, Ascard ...")     #placeholder
##stg_entry.place(x=1250, y=60)


# entry

##med_entry = Entry(root, text ="",textvar=med_v,bg="light steel blue", width=36)
# med_entry.insert(0,"Panadol, Ascard ...")     #placeholder
##med_entry.place(x=1250, y=210)



# submit function
def submit():
    print("button hit !")

    #stg = (stg_v.get())
    #med = (med_v.get())
    gender = (gender_v.get())
    age = (age_v.get())
    # family_history = (fhistory_v.get())
    # patient = (patient_v.get())
    # bpcontrol = (bpcontrol_v.get())
    headache = headache_v.get()
    breath = breath_v.get()
    visual = visual_v.get()
    nosebleeding = nosebleeding_v.get()
    diagnosed = diagnosed_v.get()
    systolic = systolic_v.get()
    diastolic = diastolic_v.get()
    #diet = diet_v.get()
    inputs = [gender,age,headache,breath,visual,nosebleeding,diagnosed,systolic,diastolic]
    '''med,stg'''
    global MDN
    global MDA
    global MDP
    MDN="Medicine: Sofvasc, Extor, Losarten"
    MDA="Medicine: Covam, Norvasc"
    MDP= "Medicine: Natrilix"
    print(inputs)
    for i in inputs:
        if i == " ":
            messagebox.showerror("Information missing","Please enter all details")
            break
        else:
            with open('checkup.csv', 'a', newline='') as f:
                write = csv.writer(f)
                write.writerow(inputs)
                f.close()
                messagebox.showinfo("showinfo", "Submitted to CSV")
                import pickle
                with open('C:\\FYP-main\\my_pickle','rb') as file:
                    pickle= pickle.load(file)

                #print(pickle.predict([[1,0,1,0,0,0,1,1,3]]))
                import numpy as np
                # R = int(input("Enter the number of rows:"))
                # C = int(input("Enter the number of columns:"))
                
                print("Enter the entries in a single line (separated by comma): ")
                
                # User input of entries in a 
                # single line separated by space
                #ck=pd.read_csv('C:\\Users\\admin\\Desktop\\Projects FL\\checkup.csv')
                #dropna(ck)
                #inputs=[1,0,1,0,0,0,1,1,3]
            # convert string list into onteger list    
                inputs = list(map(int, inputs))
                print(inputs) 
                
                # For printing the matrix
                matrix = np.array(inputs).reshape(1, 9)
                print(matrix)
                print(pickle.predict(matrix))
                global bp
                bp=pickle.predict(matrix)
                global stage1
                global stage2
                global stage3
                stage1 = 'HIGH BLOOD PRESSURE(Stage-1)'
                stage2 = 'HIGH BLOOD PRESSURE(Stage-2)'
                stage3 = 'HIGH BLOOD PRESSURE(Stage-3)'

                if bp == stage1:
                  
                    l_stg1 = Label(text=stage1,font=label_text,bg="white",fg="Orange")
                    l_stg1.place(x=1100, y=100)
                elif bp == stage2:
                    l_stg2 = Label(text=stage2,font=label_text,bg="white",fg="Orange")
                    l_stg2.place(x=1100, y=100)
                else:
                    l_stg3 = Label(text=stage3,font=label_text,bg="white",fg="Orange")
                    l_stg3.place(x=1100, y=100)
                break
                
              
    

def medicine():

   
    if bp == stage1:
                    print (MDN)
                    l_mdn = Label(text=MDN,font=label_text,bg="white",fg="Orange")
                    l_mdn.place(x=1100, y=250)
                    l_stg1 = Label(text=stage1,font=label_text,bg="white",fg="Orange")
                    l_stg1.place(x=1100, y=100)
    elif bp == stage2:
                    print (MDA)
                    l_mda = Label(text=MDA,bg="white",fg="Orange",font=label_text)
                    l_mda.place(x=1100, y=250)
                    l_stg2 = Label(text=stage2,font=label_text,bg="white",fg="Orange")
                    l_stg2.place(x=1100, y=100)
    else:
                    l_mdp = Label(text=MDP,bg="white",fg="Orange",font=label_text)
                    l_mdp.place(x=1100, y=250)
                    l_stg3 = Label(text=stage3,font=label_text,bg="white",fg="Orange")
                    l_stg3.place(x=1100, y=100)
                
  
    unchecked()
 

# Button section
# reset_button = Button(text="Reset",font="Time 12 bold",bg="white", fg="red",command=unchecked, width=12)
# reset_button.place(x=600,y=650)

submit_button = Button(text="Submit",font="Time 12 bold",bg="white",command=submit, fg="red", width=22)
submit_button.place(x=400,y=700)

# stg_button = Button(text="Stage",font="Time 12 bold",bg="white", fg="red", width=10)
# stg_button.place(x=1100,y=50)
#medi_button = Button(text="Medication",font="Time 12 bold",bg="white", fg="red", width=10)
#medi_button.place(x=1100,y=200)


#Medic
l_med = Button(text="Medication",font="Time 12 bold",bg="white",command=medicine, fg="red", width=22)
l_med.place(x=1100, y=200)



root.mainloop()


# In[ ]:





# In[ ]:




