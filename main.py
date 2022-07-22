import os
import numpy as np
from PIL import Image
import random


images_path = '../pair_face/'
without_masks_path = images_path + 'without_mask'


def save_photo(original_path, save_path):
    if os.path.exists(original_path):
        im = Image.open(original_path)
        im.save(save_path)
        return True
    return False


def get_white_masked_photo_name_by_unmasked(unmasked_name):
    return 'with-mask-default-mask-' + unmasked_name


def get_colorful_masked_name_by_unmasked(unmasked_name):
    return unmasked_name


def save_image_group(images, original_masked_path, name_conversion_func, save_path):
    with_masks_save_path = os.path.join(save_path, 'with_mask')
    without_masks_save_path = os.path.join(save_path, 'without_mask')

    if not os.path.exists(with_masks_save_path):
        os.mkdir(with_masks_save_path)

    if not os.path.exists(without_masks_save_path):
        os.mkdir(without_masks_save_path)

    for image_name in images:
        masked_name = name_conversion_func(image_name)
        masked_original_full_path = os.path.join(original_masked_path, masked_name)
        masked_full_save_path = os.path.join(with_masks_save_path, image_name)
        is_masked_saved = save_photo(masked_original_full_path, masked_full_save_path)
        if is_masked_saved is True:
            unmasked_original_full_path = os.path.join(without_masks_path, image_name)
            masked_full_save_path = os.path.join(without_masks_save_path, image_name)
            save_photo(unmasked_original_full_path, masked_full_save_path)


def pre_process_masks(masks_path, new_path, name_conversion_func):
    images = os.listdir(without_masks_path)
    random.shuffle(images)
    train, validate, test = np.split(images, [int(len(images) * 0.7), int(len(images) * 0.85)])

    train_data = [
        new_path + 'train/',  # save_path
        train  # data
    ]
    validate_data = [
        new_path + 'validate/',  # save_path
        validate  # data
    ]

    test_data = [
        new_path + 'test/',  # save_path
        test  # data
    ]

    all_groups = [train_data, validate_data, test_data]

    for group in all_groups:
        save_path, data = group
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_image_group(data, masks_path, name_conversion_func, save_path)


def main():
    base_division_path = '../pair_face_divided/'
    white_paths = [
        images_path + 'with_white_masks/',  # original_path
        base_division_path + 'white_masks/',  # save_path
        get_white_masked_photo_name_by_unmasked  # conversion_func
    ]
    black_paths = [
        images_path + 'with_black_mask/',  # original_path
        base_division_path + 'black_masks/',  # save_path
        get_colorful_masked_name_by_unmasked  # conversion_func
    ]
    blue_paths = [
        images_path + 'with_blue_masks_full/',  # original_path
        base_division_path + 'blue_masks/',  # save_path
        get_colorful_masked_name_by_unmasked  # conversion_func
    ]

    all_colors = [white_paths, black_paths, blue_paths]

    for color in all_colors:
        original_path, save_path, conversion_func = color
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        pre_process_masks(original_path, save_path, conversion_func)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
