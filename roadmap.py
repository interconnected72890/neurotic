roadmap = [
    ["Create a Class which Allows for Model Insertion and simple Function Calls for Training and Saving of Results", 0],
    ["Create a function which allows for UPDATES and NO UPDATES -> Tests Accuracy as you go", 0],
    ["Create a Cool Verbosity of Progress -> Create a Function which calls that Specific Line", 0],
    ["Create a Function which will allow to print status update; Or Save values to File", 0],
    ["Figure out what In_Channels & Out_Channels are", 0],
    ["Download Datasets_old For Research", 2],
    ["Turn Metro Dataset Into Workable Dataset -> Create Class For it with Callable Function to Extract", 1],
    ["Turn Tetouan Dataset Into Workable Dataset -> Create Class For it with Callable Function to Extract", 0],
    ["Turn Steel Dataset Into Workable Dataset -> Create Class For it with Callable Function to Extract", 0]
]


for x in roadmap:
    if len(x[0]) > 0:
        if x[1] == 0:
            print("\033[31m[NOT STARTED] \033[0m", end="")
        elif x[1] == 1:
            print("\033[33m[IN PROGRESS] \033[0m", end="")
        else:
            print("\033[32m[ COMPLETED ] \033[0m", end="")
        print(x[0])


















