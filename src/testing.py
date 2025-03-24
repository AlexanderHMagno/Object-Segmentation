# Provide paths to your test images
import os
import random

def get_image_names():
    # Get people images
    people_dir = "public/images/people" 
    # Get background images
    background_dir = "public/images/backgrounds"


    data = [
      ['e.jpg', 'e.png', 10, 60],
      ['f.jpg', 'f.png', 10, 60],
      ['g.jpeg', 'g.png', 10, 40],      
      ['a.jpg', 'a.png', 10, 130],
      ['b.jpg', 'b.png', 10, 40],
      ['c.jpg', 'c.png', 10, 60],
      ['d.png', 'd.png', 10, 50]
    ]

    generate_testing_list = []

    for i in range(len(data)):
      generate_testing_list.append(
        list((
          os.path.join(people_dir, data[i][0]), 
          os.path.join(background_dir, data[i][1]), 
          data[i][2], 
          data[i][3]
          ))
        )
    
    return generate_testing_list




