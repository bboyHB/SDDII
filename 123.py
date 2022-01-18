from PIL import Image
a = Image.open('datasets/RSDDs1_cycle/testA/rail_12_0.jpg')
b = a.crop((0, 0, a.width-1, a.height))
pass