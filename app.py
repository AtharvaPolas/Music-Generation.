import streamlit as st
import numpy as np
import pickle
import random
from tensorflow.keras.models import load_model
from music21 import instrument, note, chord, stream

st.title("🎵 AI Music Generator")
st.write("Generate piano music using a Deep Learning model")
temperature = st.slider("Creativity", 0.5, 10.0, 2.0)


# Load trained model
model = load_model("model.h5")

# Load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))


# Temperature sampling function (same as Colab)
def sample_with_temperature(preds, temperature=2):

    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas)


# Generate music
def generate_music(temperature):

    sequence_length = 100
    network_input = []

    last_note = None

    for i in range(0, len(notes) - sequence_length):

        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[n] for n in sequence_in])

    network_input = np.reshape(
        network_input,
        (len(network_input), sequence_length, 1)
    )

    start = np.random.randint(0, len(network_input)-1)

    pattern = network_input[start].copy()

    prediction_output = []

    for note_index in range(20):

        prediction_input = np.reshape(pattern, (1, len(pattern), 1))

        prediction = model.predict(prediction_input, verbose=0)

        index = sample_with_temperature(prediction[0], temperature)
        result = pitchnames[index]

        # prevent repeating the same note
        if result == last_note:
            index = sample_with_temperature(prediction[0], temperature)
            result = pitchnames[index]

        last_note = result

        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


# Convert generated notes to MIDI
def create_midi(prediction_output):

    offset = 0
    output_notes = []

    for pattern in prediction_output:

        if ('.' in pattern) or pattern.isdigit():

            notes_in_chord = pattern.split('.')
            notes_list = []

            for current_note in notes_in_chord:

                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes_list.append(new_note)

            new_chord = chord.Chord(notes_list)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:

            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Random rhythm like Colab
        offset += random.choice([0.25, 0.5, 0.75])

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp="generated_music.mid")


# Streamlit button
if st.button("Generate Music 🎼"):

    st.write("Generating music...")

    prediction_output = generate_music(temperature)

    create_midi(prediction_output)

    st.success("Music Generated!")

    with open("generated_music.mid", "rb") as file:

        st.download_button(
            label="Download Music",
            data=file,
            file_name="ai_music.mid",
            mime="audio/midi"
        )













# import streamlit as st
# import numpy as np
# import pickle
# from tensorflow.keras.models import load_model
# from music21 import instrument, note, chord, stream

# st.title("🎵 AI Music Generator")

# st.write("Generate piano music using a Deep Learning model")

# # Load model and notes
# model = load_model("model.h5")

# with open("notes.pkl", "rb") as f:
#     notes = pickle.load(f)

# pitchnames = sorted(set(notes))
# note_to_int = dict((note, number) for number, note in enumerate(pitchnames))


# # Generate function
# def generate_music():

#     sequence_length = 100

#     network_input = []

#     for i in range(0, len(notes) - sequence_length):
#         sequence_in = notes[i:i + sequence_length]
#         network_input.append([note_to_int[n] for n in sequence_in])

#     network_input = np.reshape(network_input,
#                                (len(network_input), sequence_length, 1))

#     start = np.random.randint(0, len(network_input)-1)

#     pattern = network_input[start]
#     prediction_output = []

#     for note_index in range(100):

#         prediction_input = np.reshape(pattern, (1, len(pattern), 1))
#         prediction = model.predict(prediction_input, verbose=0)

#         index = np.argmax(prediction)
#         result = pitchnames[index]

#         prediction_output.append(result)

#         pattern = np.append(pattern, index)
#         pattern = pattern[1:len(pattern)]

#     return prediction_output


# # Convert notes to MIDI
# def create_midi(prediction_output):

#     offset = 0
#     output_notes = []

#     for pattern in prediction_output:

#         if ('.' in pattern) or pattern.isdigit():

#             notes_in_chord = pattern.split('.')
#             notes_list = []

#             for current_note in notes_in_chord:
#                 new_note = note.Note(int(current_note))
#                 new_note.storedInstrument = instrument.Piano()
#                 notes_list.append(new_note)

#             new_chord = chord.Chord(notes_list)
#             new_chord.offset = offset
#             output_notes.append(new_chord)

#         else:

#             new_note = note.Note(pattern)
#             new_note.offset = offset
#             new_note.storedInstrument = instrument.Piano()
#             output_notes.append(new_note)

#         offset += 0.5

#     midi_stream = stream.Stream(output_notes)
#     midi_stream.write('midi', fp="generated_music.mid")


# if st.button("Generate Music 🎼"):

#     st.write("Generating music...")

#     prediction_output = generate_music()

#     create_midi(prediction_output)

#     st.success("Music Generated!")

#     with open("generated_music.mid", "rb") as file:
#         st.download_button(
#             label="Download Music",
#             data=file,
#             file_name="ai_music.mid",
#             mime="audio/midi"
#         )