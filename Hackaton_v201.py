# This is the first code file for Hackathon.
# The date is 1st-July-2024
# We can use following models:
    # Whisper -> Open AI
    # M4t -> Meta
    # wav2vec -> Facebook

# The only differnet in version 2.1 is it is implemented using streamlit

#-----------------------------------------------------------------#
# Imports
import datetime
import os
import pyaudio
import wave 
import numpy as np
import pandas as pd
import keyboard
import streamlit as st

from thefuzz import process

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#-----------------------------------------------------------------#

"""
We will first recored audio and store this. The stored audio will then 
be used as input to ASR. The model will transcribe names. 
"""

#  We will use pyaudio to get audio stream from terminal

def add_names(current_dir, complete_list_dir, complete_list_file, new_name):
    """
    This function is used to manullay add names of student to original list.

    input: Original Student list
    Output: Final Updated Student list
    """ 

    combined_file = os.path.join(current_dir, complete_list_dir, complete_list_file)
    df_student = pd.read_csv(combined_file)

    df_student.loc[len(df_student.index)] = [new_name] 
    df_student.to_csv(combined_file, index = False)

    st.success(f"Student list updated with {new_name}")

def remove_names(current_dir, complete_list_dir, complete_list_file, name_to_remove):
    """
    This function removes a student's name from the original list.

    input: Name to be removed
    Output: Final Updated Student list
    """ 
    combined_file = os.path.join(current_dir, complete_list_dir, complete_list_file)
    df_student = pd.read_csv(combined_file)
    
    if name_to_remove in df_student['Name'].values:
        df_student = df_student[df_student['Name'] != name_to_remove]
        df_student.to_csv(combined_file, index=False)
        st.success(f"{name_to_remove} has been removed from the student list.")
    else:
        st.warning(f"{name_to_remove} not found in the student list.")

        
def get_timestamp():

    # https://www.geeksforgeeks.org/get-current-timestamp-using-python/
    # https://stackoverflow.com/questions/3961581/in-python-how-to-display-current-time-in-readable-format?noredirect=1&lq=1
    
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y-%m-%d %H-%M-%S")

    return time_stamp


def audio_input(current_dir, input_dir):
    """
    This function takes input audio stream for processing. 
    The input stream results are stored for future usage
    and checking with timestamps.

    # Pyaudio for input using microphone https://people.csail.mit.edu/hubert/pyaudio/

    Input: Audio from Microphone
    Output: Saved Audio file path
    """
    combined_dir = os.path.join(current_dir,input_dir)

    if os.path.exists(combined_dir):
        pass
    else:
        os.mkdir(combined_dir)


    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 
    RATE = 44100
    RECORD_SECONDS = 5

    time_stamp = get_timestamp()
    input_file = "audio_" + time_stamp + ".wav"
    input_path = os.path.join(input_dir, input_file)

    with wave.open(input_path, 'wb') as wf:
        audio_input = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio_input.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = audio_input.open(
                        format=FORMAT, 
                        channels=CHANNELS, 
                        rate=RATE, 
                        input=True)
       
        recording = False
        while True:
            if keyboard.is_pressed('p') and not recording:
                print('-'*80)
                st.write('Recording...')
                recording = True
            elif keyboard.is_pressed('q') and recording:
                st.write('Recording stopped.')
                break
            if recording:
                wf.writeframes(stream.read(CHUNK))

    stream.close()
    audio_input.terminate()

    return input_path


def transcription(audio_file_path):
    """
    This function takes input the audio file and returns the transcribed text.
    The model uses OpenAI's Whisper model for Automatic Speech Recognition.
    The problem we have to deal with is to properly transcribe names other than English
    names.

    # https://huggingface.co/openai/whisper-large-v3
    
    Input: Audio file path
    Output: Transcribed audio
    
    """
    device =  "cpu"

    model_id = "openai/whisper-tiny"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        device=device,
    )

    result = pipe(audio_file_path, generate_kwargs={"language": "english"}, return_timestamps=False)
    st.write(result["text"])

    return result['text']

def parsing_transcription(transcription):
    """
    This function is used to parse the transcribed text obtained from transcription model.
    We parse the text so that each name is in a differnet row.
    This will later be used to match names from student list.

    Input: Transcription String
    output: Numpy Array with all names as rows
    """

    names_list = transcription.split()
    names_array = np.array(names_list)

    return names_array


def Attendance(current_dir, output_dir, complete_list_file, complete_list_dir, names_list):
    """
    This function matches names obtained from transcription with student list.
    Attendance is marked for available students.
    The output is stored in a csv file
    
    # https://spotintelligence.com/2023/07/10/name-matching-algorithm/
    # https://www.datacamp.com/tutorial/fuzzy-string-python
    # https://medium.com/@ammubharatram/fuzzy-name-matching-comparing-customer-names-with-watchlist-entities-as-part-of-name-sanctions-fed922b3f772
    # https://github.com/maladeep/Name-Matching-In-Python/blob/master/Surprisingly%20Effective%20Way%20To%20Name%20Matching%20In%20Python.ipynb

    # https://github.com/Christopher-Thornton/hmni
    # https://github.com/seatgeek/thefuzz


    Input: Transcribed student list & Overall student list
    Output: .csv file with student attendance
    """

    # load complete_list
    combined_file = os.path.join(current_dir, complete_list_dir, complete_list_file)
    df_student = pd.read_csv(combined_file)
    # convert this to a list
    student_list = df_student.values.tolist()
    print(student_list)

    # Match Names using thefuzz library
    # Store values in a dataframe with name and score or could also have a master file which is updated daily with attendance.
    df_Attendance =  df_student.copy()
    df_Attendance['Attendance'] = "Absent"
    #df_Attendance['Score'] = ""


    for name in names_list:
        match = process.extract(name, student_list, limit = 1 )
        print(match)
        name = match[0][0][0]
        score = match[0][1]
        print(f"name: {name}, score: {score}")

        
        row_id = df_Attendance.index[df_Attendance['Name'] == name]
        if not row_id.empty:
            row_index = row_id[0]  # Get the first matching index
            df_Attendance.at[row_index, 'Attendance'] = 'Present'
            #df_Attendance.at[row_index, 'Score'] = score

    # Save Attendance
    combined_dir_output = os.path.join(current_dir, output_dir)
    if os.path.exists(combined_dir_output):
        pass
    else:
        os.mkdir(combined_dir_output)

    # Save filename with time stamps
    time_stamp =get_timestamp()
    output_file = 'Attendance_' + time_stamp + '.csv'
    output_path = os.path.join(output_dir, output_file)
    df_Attendance.to_csv(output_path, index = False)

    st.success(f"Attendance saved to {output_path}")

    return df_Attendance
    
    #--------------------------------------------------------------------------------#