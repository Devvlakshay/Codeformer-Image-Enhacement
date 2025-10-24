from PIL import Image

img = Image.open("/Users/lakshyaborasi/Desktop/CodeFormer/test_images/kyc_9f2862646a668d22_1737891157687_faceverify.jpg")
img = img.resize((512, 512))
img.save("/Users/lakshyaborasi/Desktop/CodeFormer/test_images/hi_512.png")
print(" Resized image saved as faceverify_H7rkAiRZZg_512.png")