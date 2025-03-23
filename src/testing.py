# Provide paths to your test images
import os
import random

def get_image_names():
    # Get background images
    background_dir = "public/images/backgrounds"
    background_images = [f for f in os.listdir(background_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Get people images
    people_dir = "public/images/people" 
    people_images = [f for f in os.listdir(people_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    

    generate_testing_list = []

    random.shuffle(background_images)
    random.shuffle(people_images)

    min_length = min(len(background_images), len(people_images))

    # Get 10 random pairs
    for i in range(min_length):
        background = background_images[i]
        person = people_images[i]
        generate_testing_list.append(list(( os.path.join(people_dir, person), os.path.join(background_dir, background), 10, 100)))

    
    return generate_testing_list




