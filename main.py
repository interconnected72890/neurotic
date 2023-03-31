import torch
import torchvision
import matplotlib.pyplot as plt
import time


device = "cuda" if torch.cuda.is_available() else "cpu"


model = torch.nn.Sequential(

    torch.nn.Conv2d(kernel_size=(3, 3), in_channels=1, out_channels=16),
    torch.nn.LeakyReLU(),
    torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),

    torch.nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=8),
    torch.nn.LeakyReLU(),
    torch.nn.MaxPool2d(kernel_size=(2, 2), padding=1),

    torch.nn.Flatten(),

    torch.nn.Linear(in_features=1152, out_features=400),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(in_features=400, out_features=500),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(in_features=500, out_features=10),
    torch.nn.LogSoftmax(dim=1)

)

model = model.to(device)


batch_size = 128

# Get Data
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,)).to(device),])

mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)


n_iters = 1000
num_epochs = int(n_iters / (len(mnist_trainset) / batch_size))


# Get Model
# model = FNN().to(device)


optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss().to(device)


print("GPU Available:   " + str(torch.cuda.is_available()))
print("GPU Initialized: " + str(torch.cuda.is_initialized()))
print("Device Used:     " + device.upper())
print("Device Name:     " + torch.cuda.get_device_name())

start = time.time()

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Load images with gradient accumulation capabilities

        images = images.requires_grad_()

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = loss_func(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        # TODO Create a function which allows for UPDATES and NO UPDATES -> Tests Accuracy as you go

        if iter % 500 == 0:
            print("#", end="")
            # TODO Create a Cool Verbosity of Progress -> Create a Function which calls that Specific Line
            """
            Example:
            Iteration: 938. Loss: 3.971845865249634. Accuracy: 98.33999633789062
            Iteration: 938. Loss: 3.971845865249634. Accuracy: 98.33999633789062
            ...
            ...
            [############..........................................................] 
            """

print("\n{}".format(time.time() - start))









# TODO Turn Below Code into a Function
# Create a Function which will allow to print status update; Or Save values to File
"""
This will allow you to save final status to file

This will also allow you to print status updates DURING training -> So A boolean value or something to differentiate


def get_status(save: bool = False, 



"""
with torch.no_grad():

    correct = 0
    total = 0
    loss = 0.0

    for images_test, labels_test in test_loader:
        # Load images with gradient accumulation capabilities

        images_test = images_test.to(device)
        labels_test = labels_test.to(device)

        images_test = images_test.requires_grad_()

        # Forward pass only to get logits/output
        outputs = model(images_test)

        loss += loss_func(outputs, labels_test)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels_test.size(0)

        # Total correct predictions
        correct += (predicted == labels_test).sum()

    accuracy = 100 * correct / total  # Average the Accuracy over the Batches
    loss /= total  # Average the Loss over the Batches

    # Print Loss
    print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))


torch.save(model.state_dict(), "./saved_model")
