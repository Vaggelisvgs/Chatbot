from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from tkinter import *
import nltk
import random
import json
import pickle
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')

# create the train data for the neuron network
train_data = []

#load the json file
json_data = json.loads(open('Data\objects.json').read())

#write the no value char like ! and ?
no_value_char = ['!', '?']

#initallize the lists for the data
word_data, class_data, docs_data = [], [], []

for temp_data in json_data['objects']:
    for k in temp_data['user_text']:

        # token each word of user text from json file
        token_word = nltk.word_tokenize(k)

        #incrise the  length of the list word_data by the number of tokend words
        word_data.extend(token_word)

        # add document data in the corpus
        docs_data.append((token_word, temp_data['subject']))

        # add to our class data the subjects of all objects list
        if temp_data['subject'] not in class_data:
            class_data.append(temp_data['subject'])

# lemmaztize and lower each word and remove the possible duplicates
lem_data = WordNetLemmatizer()

#sorted new word to cut the duplicates
new_words = [lem_data.lemmatize(w.lower()) for w in word_data if w not in no_value_char]
new_words = sorted(list(set(new_words)))

# sort classes
new_class = sorted(list(set(class_data)))

#print the value of documents, documents = combination between patterns and intents
#print the number of intents
#print the words of vocabulary
print(len(docs_data), "number of documents"+'\n'+str(len(new_class)), "number of class", new_class ,'\n'+str(len(new_words)), "number of words of vocabulary", new_words)

#serializes the words and the classes and return the bytes object of the serialized objects
pickle.dump(new_words, open('words.pkl', 'wb'))
pickle.dump(new_class, open('class.pkl', 'wb'))

# create an empty array for our output with root the len of the new sorted classes
empty_output = [0] * len(new_class)

# train the set and store the total words for each sentence
for doc in docs_data:
    # initialize our total words
    total_words = []
    # list of token words for the pattern of the first element of documents
    pat_for_words = doc[0]

    # lem each word - create total words in attempt to represent the related words
    for every in pat_for_words :
        pat_for_words = [lem_data.lemmatize(every.lower())]

    # create our total words array with 1, if word match found in current pattern
    for w in new_words:
        if w in pat_for_words:
            total_words.append(1)
        else:
            total_words.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    sentence_output = list(empty_output)
    sentence_output[new_class.index(doc[1])] = 1

    train_data.append([total_words, sentence_output])

# shuffle our features and turn into np.array
random.shuffle(train_data)
training = np.array(train_data)
# create train and test lists.
train_patterns = list(training[:, 0])
train_intents = list(training[:, 1])
print("Successful creation of training data!")

# Create model with 3 layers.
# First layer 128 neurons
# second layer 64 neurons
# 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_patterns[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_intents[0]), activation='softmax'))

# Compile model
# Stochastic Gradient Descent (SGD) with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting the model
history = model.fit(np.array(train_patterns), np.array(train_intents), epochs=200, batch_size=5, verbose=1)

#saving the model
model.save('chatbot_model.h5', history)

#load the saved model
model = load_model('chatbot_model.h5')

#load the saved json
intents = json.loads(open('Data/objects.json').read())

#load the words data
words = pickle.load(open('words.pkl', 'rb'))

#load the class data
classes = pickle.load(open('class.pkl', 'rb'))

#text preprocessing of data and predict the class
def text_preprocessing_predict_class(sent, model):
    # tokenize the pattern - split words into array
    sent_for_words = nltk.word_tokenize(sent)

    # stem each word - create short form for word
    for every in sent_for_words:
        sent_for_words = [lem_data.lemmatize(every.lower())]

    # bag of words - matrix of N words, vocabulary matrix
    bag_of_words = [0] * len(words)

    # for every word in the sentence
    for word in sent_for_words:
        for first, second in enumerate(words):
            if second == word:
                # assign 1 if current word is in the vocabulary position
                bag_of_words[first] = 1
    # filter out predictions below a threshold
    bakc_array = np.array(bag_of_words)
    result = model.predict(np.array([bakc_array]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Creating GUI with tkinter
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatWindow.config(state=NORMAL)
        ChatWindow.insert(END, "You: " + msg + '\n\n')
        ChatWindow.config(foreground="#442265", font=("Verdana", 12))

        ints = text_preprocessing_predict_class(msg, model)
        tag = ints[0]['intent']
        list_of_intents = intents['objects']
        for i in list_of_intents:
            if (i['subject'] == tag):
                result = random.choice(i['bot_text'])
                break
        res = result
        ChatWindow.insert(END, "Bot: " + res + '\n\n')

        ChatWindow.config(state=DISABLED)
        ChatWindow.yview(END)

root = Tk()
root.title("Direct communication for COVID-19")
root.geometry("600x400")
root.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatWindow = Text(root, bd=0, bg="white", height="80", width="500", font="Times", )

ChatWindow.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatWindow.yview, cursor="star")
ChatWindow['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(root, font=("Times", 12, 'bold'), text="Send", width="12", height=5,
                    bd=2, bg="#00802b", activebackground="#00e64d", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(root, bd=0, bg="#a3c2c2", width="290", height="50", font="Times")

# Place all components on the screen
scrollbar.place(x=576, y=6, height=270)
ChatWindow.place(x=6, y=6, height=270, width=570)
EntryBox.place(x=128, y=301, height=90, width=450)
SendButton.place(x=6, y=301, height=90)

root.mainloop()