from unittest.mock import sentinel
from bot import telegram_chatbot
# import gizoogle
import configparse as cfg
import random
import json

import torch 

from model import NeuralNet
from nltk_utils import bag_of_words, tokeniz
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# bot_name = "Sam"
# print("Let's chat! (type 'quit' to exit)")

def maker(mess):
    sentence = mess
    if sentence == "quit":
      # break
      
      return 
    if sentence =="order":
      reply = "yes wait we are ordering !"
      # break
      return reply
    sentence = tokeniz(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                reply = f" {random.choice(intent['responses'])}"
                return reply
    else:
        reply = "I do not understand..."
        return reply



# telegram

bot = telegram_chatbot()

# def make_reply(msg):
#     reply = None
#     if msg is not None:
        
#         reply = msg
#         if "hi" in reply:
#          reply = "how are you bruh"
#          return reply  
#         else : 
#          return reply

update_id = None
while True:
    
    updates = bot.get_updates(offset=update_id)
    updates = updates["result"]
    if updates:
        for item in updates:
            update_id = item["update_id"]
            try:
                message = str(item["message"]["text"])
            except:
                message = None
            from_ = item["message"]["from"]["id"]
            # reply = make_reply(message)
            reply =maker(message)
            bot.send_message(reply, from_)