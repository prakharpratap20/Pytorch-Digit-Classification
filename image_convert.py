from PIL import Image
from torchvision.transforms import ToTensor, Grayscale

# Open the image 
img = Image.open('four.png') # enter the name of image you want to convert 

# Convert the image to grayscale (if not already in grayscale)
img = Grayscale()(img)

# Resize the image to 28x28
img = img.resize((28, 28))

# Convert the image to a PyTorch tensor
img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

# Save the preprocessed image
img_path = 'four_img.png' # name of image after getting converted
img.save(img_path)

