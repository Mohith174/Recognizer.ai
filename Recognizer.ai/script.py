import json
from parse import Parse
from dbman import dbman
from m_l.CNN import Recognizer_ai

with open('config.json', 'r') as configurator: 
    config = json.load(configurator)
    
'''
csv_path = config['csv']
dataman = dbman(config['dbpath'])
parse = Parse(config['adir'], config['odir'],dataman,csv_path)
parse.processAudio()
dataman.close()
'''
def main_menu():
    csv_path = config['csv']
    dataman = dbman(config['dbpath'])
    parse = Parse(config['adir'], config['odir'], dataman, csv_path)
    
    cnn_model = Recognizer_ai(config['dbpath'])

    while True:
        print("\n--- Main Menu ---")
        print("1: Process Audio")
        print("2: Train CNN Model")
        print("3: Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            parse.processAudio()
            dataman.close()
        elif choice == '2':
            model, history = cnn_model.train()
            print("Model trained. History of training:", history.history)
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please choose again.")

if __name__ == "__main__":
    main_menu()