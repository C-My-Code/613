1. In order to use the GPU functions in this library images must be converted to greyscale and stored in a binary file. A script img_to_bin.py has been provided that does this.
When calling this script simply pass in the filenames you wish to convert. This script can process multiple images in a single call. For example: "python3 image_to_bin.py img1.jpg img2.png"

2. Another script bin_to_image.py has been provided which converts the processed binary image data back to images. To use this script pass in the dimensions of each image contained
in the binary image data. For example: "python3 bin_to_image.py 800 800" to retreive a single 800x800 image from the data. "python3 bin_to_image.py 800 800 800 800" would be used to
retreive two 800x800 images. Also, note the filname the script is looking for "image_out.bin".

3. Parameters for each GPU libary function are noted in the header file img_lib.cuh

5. To use the performance test example build with peft_test.cu and call your output file with the number of times each function should execute each image size for average performance capture.
For example: "./perf_test 50" to process each image size 50 times
