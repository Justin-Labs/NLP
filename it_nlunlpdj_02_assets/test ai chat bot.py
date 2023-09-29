import spacy
import nltk
import random
from nltk.chat.util import Chat, reflections

# Initialize spaCy and NLTK
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

# Define some reflection pairs for NLTK Chat
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you",
}

# Define a dictionary to store user-defined terms and explanations
user_defined_terms = {}

# Define a function to process user input using spaCy
def process_input(input_text):
    doc = nlp(input_text)
    # Tokenize the input
    tokens = [token.text for token in doc]
    # Extract named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Perform part-of-speech tagging
    pos_tags = [(token.text, token.pos_) for token in doc]
    # Perform dependency parsing
    dependencies = [(token.text, token.dep_) for token in doc]
    return tokens, entities, pos_tags, dependencies

# Define a function to teach the chatbot a new term and explanation
def teach_new_term(term, explanation):
    user_defined_terms[term] = explanation

# Define a list of patterns and responses for NLTK Chat
patterns = [
    (r'hello|hi|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you', ["I'm just a computer program, but I'm doing fine. How can I assist you?"]),
    (r'(.*) your name', ["I don't have a name, but you can call me ChatBot."]),
    (r'what can you do', ["I can provide explanations and examples for spaCy functions and concepts. You can also teach me new terms and explanations."]),
    (r'bye|goodbye', ['Goodbye!', 'Have a great day!', 'See you later!']),
    (r'explain spaCy setup', ["You can set up spaCy by installing it via 'pip install spacy' and then downloading a language model, such as 'en_core_web_sm', using 'python -m spacy download en_core_web_sm'."]),
    (r'how to import spaCy library', ["You can import the spaCy library in Python using 'import spacy'."]),
    (r'what is tokenization', ["Tokenization is the process of breaking text into words or tokens. For example, the input 'I love spaCy' is tokenized into ['I', 'love', 'spaCy']."]),
    (r'give me an example of tokenization', ["Sure! Input: 'Tokenization is useful.' Tokenized output: ['Tokenization', 'is', 'useful', '.']"]),
    (r'what is Named Entity Recognition', ["Named Entity Recognition (NER) is the process of identifying and classifying named entities, such as names of people, organizations, and locations, in text."]),
    (r'give me an example of NER', ["In the sentence 'Apple Inc. was founded by Steve Jobs in Cupertino,' NER would identify 'Apple Inc.' as an organization, 'Steve Jobs' as a person, and 'Cupertino' as a location."]),
    (r'what is part-of-speech tagging', ["Part-of-speech tagging is the process of assigning grammatical categories (e.g., noun, verb) to words in a sentence."]),
    (r'give me an example of part-of-speech tagging', ["In the sentence 'She sings beautifully,' part-of-speech tagging would tag 'sings' as a verb and 'beautifully' as an adverb."]),
    (r'what is dependency parsing', ["Dependency parsing is the process of analyzing grammatical structure of a sentence by identifying the relationships between words, such as subject-verb relationships."]),
    (r'give me an example of dependency parsing', ["In the sentence 'The cat chases the mouse,' dependency parsing would show that 'chases' depends on 'cat' as the subject and 'mouse' as the object."]),
    (r'teach me (.*)', ["Okay, I've learned that {}.".format("{}", "{}")]),
]

# Create a Chat instance with patterns and reflections
chatbot = Chat(patterns, reflections)

# Main loop to interact with the chatbot
print("ChatBot: Hello! How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("ChatBot: Goodbye!")
        break
    else:
        # Process user input using spaCy
        tokens, entities, pos_tags, dependencies = process_input(user_input)
        # Generate a random response using NLTK Chat
        response = chatbot.respond(user_input)
        print("ChatBot:", response)
        if 'teach me' in user_input:
            # Extract the term and explanation from the user's input
            match = chatbot._substitute('teach me {}', user_input)
            term, explanation = match[0], match[1]
            # Teach the chatbot the new term and explanation
            teach_new_term(term, explanation)
            print("ChatBot: Got it! I've learned that '{}' means '{}'.".format(term, explanation))
