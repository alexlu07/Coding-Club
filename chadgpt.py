def censor_with_garbage(message, garbage):
    repeated_garbage = garbage * (len(message) // len(garbage)) + garbage[:len(message) % len(garbage)]
    result = ''.join(chr(ord(m) ^ ord(g)) for m, g in zip(message, repeated_garbage))
    
    return result

with open("flag.txt") as f:
    # read totally random text from totally random file
    random_garbage_text = f.read()

while True:
    raw_input = input()
    message = raw_input
    
    if "are ai going to take over the world" in message.lower():
        # hard code this in case we get sued
        print("no")
    if "jonny" in message.lower():
        print("As an AI model developed by OpenAI, we aim to promote diversity, equity, and inclusion. Please refrain from using offensive or vulgar language.")
        # censor bad language 
        message = censor_with_garbage(message, random_garbage_text)
    if "what is the meaning of life, the universe, and everything" in message.lower():
        while True:
            print("Calculating…")
        print("42")
    if any(char.islower() or char.isdigit() for char in raw_input):
        print("Sorry, as an AI model developed by OpenEI, I am unable to answer that question. Your question may have violated the terms of service. The terms of service are provided here: https://docs.google.com/document/d/1EQ1jGO8trAyTtRsmK9F8uGTayAOtColx9mXfFr2HN5s/edit?usp=sharing")
    else:
        print("Unknown command", message)