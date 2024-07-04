# Streamlit App
import streamlit as st
import os
import pandas as pd
from Hackaton_v201 import add_names, audio_input, transcription, parsing_transcription, Attendance, remove_names

# Streamlit App
st.title("Automatic Attendance System")


current_dir = os.getcwd()
input_dir = "Audio Inputs"
output_dir = "Student Attendance"
complete_list_dir = "Student List"
pictures_dir = 'pictures'
complete_list_file = "Complete_Student_List.csv"
logo_pic = "Logo.jpg"
first_pic = "Attendance_1.jpg"


logo_ = os.path.join(current_dir, pictures_dir, logo_pic)
first_pic_ = os.path.join(current_dir, pictures_dir, first_pic)

st.logo(logo_)


# Sidebar options
option = st.sidebar.selectbox("Options", ["Mark Attendance", "Previous Records", "Edit Student List"])

if option == "Mark Attendance":
    st.image(first_pic_, width = 500)
    st.subheader("Complete Student List")
    if os.path.exists(os.path.join(current_dir, complete_list_dir, complete_list_file)):
        df_student_list = pd.read_csv(os.path.join(current_dir, complete_list_dir, complete_list_file))
        st.dataframe(df_student_list, hide_index=True)  # Displaying the student list without index
    else:
        st.warning("Student list file not found.")

    # Mark attendance end-to-end
    st.subheader("Attendance")
    if st.button("Mark Attendance"):
        st.write('Press "p" to start recording, and "q" to stop recording.')
        audio_file_path = audio_input(current_dir, input_dir)
        st.success(f"Audio recorded and saved at {audio_file_path}")

        transcribed_text = transcription(audio_file_path)
        st.text_area("Transcribed Text", transcribed_text)

        names_list = parsing_transcription(transcribed_text)
        df_attendance = Attendance(current_dir, output_dir, complete_list_file, complete_list_dir, names_list)

        st.subheader("Marked Attendance")
        st.dataframe(df_attendance, hide_index=True)  # Displaying the attendance without index

elif option == "Previous Records":
    st.subheader("Previous Audio Records")
    audio_files = [f for f in os.listdir(os.path.join(current_dir, input_dir)) if f.endswith('.wav')]
    if audio_files:
        selected_audio = st.selectbox("Select an audio file", audio_files)
        audio_path = os.path.join(current_dir, input_dir, selected_audio)
        st.audio(audio_path, format='audio/wav')
    else:
        st.warning("No audio records found.")

    st.subheader("Previous Attendance Records")
    attendance_files = [f for f in os.listdir(os.path.join(current_dir, output_dir)) if f.endswith('.csv')]
    if attendance_files:
        selected_attendance = st.selectbox("Select an attendance file", attendance_files)
        attendance_path = os.path.join(current_dir, output_dir, selected_attendance)
        df_attendance = pd.read_csv(attendance_path)
        st.dataframe(df_attendance, hide_index=True)  # Displaying the attendance records without index
    else:
        st.warning("No attendance records found.")

elif option == "Edit Student List":
    st.subheader("Add New Student")
    new_student_name = st.text_input("Enter new student name")
    if st.button("Add Student"):
        if new_student_name:
            add_names(current_dir, complete_list_dir, complete_list_file, new_student_name)
        else:
            st.warning("Please enter a name before adding.")

        df_student_list = pd.read_csv(os.path.join(current_dir, complete_list_dir, complete_list_file))
        st.dataframe(df_student_list, hide_index=True)  # Displaying the student list without index
        
    st.subheader("Remove Student")
    remove_student_name = st.text_input("Enter student name to remove")
    if st.button("Remove Student"):
        if remove_student_name:
            remove_names(current_dir, complete_list_dir, complete_list_file, remove_student_name)
        else:
            st.warning("Please enter a name before removing.")

        df_student_list = pd.read_csv(os.path.join(current_dir, complete_list_dir, complete_list_file))
        st.table(df_student_list)  # Displaying the student list without index

# Handling navigation errors
else:
    st.error("Invalid option selected. Please choose from the available options.")