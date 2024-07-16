from PIL import Image
import os



def crop_large_images(input_folder, output_folder, crop_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".png"):
            input_image_path = os.path.join(input_folder, file)
            img = Image.open(input_image_path)
            width, height = img.size
            count = 0

            for i in range(0, width - crop_size + 1, crop_size):
                for j in range(0, height - crop_size + 1, crop_size):
                    box = (i, j, i + crop_size, j + crop_size)
                    cropped_img = img.crop(box)

                    output_filename = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_output_{count}.png")
                    cropped_img.save(output_filename)
                    count += 1

if __name__ == "__main__":

    input_folder = "test_input"
    output_folder = "test_output"

    # input_folder = "input_image"
    # output_folder = "cut_image"
    crop_size = 256

    crop_large_images(input_folder, output_folder, crop_size)