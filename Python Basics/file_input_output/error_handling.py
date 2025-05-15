#file handling: error handling

try:
    with open('non_existent_file.txt', 'r') as file:
        content = file.read()

except FileNotFoundError:
    print("The file doesn't exist.")