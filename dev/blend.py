from typing import List
import cv2

from spg.popos import InputImage


# get weight value based on liner interpolation
# def blend_weight_calculator(left_image_idx, right_image_idx, current_image_idx):
#     window_width = right_image_idx - left_image_idx
#
#     if window_width > 0:
#         left_weight = abs(right_image_idx - current_image_idx) / window_width
#
#         right_weight = abs(current_image_idx - left_image_idx) / window_width
#     else:
#         left_weight = 0.5
#         right_weight = 0.5
#
#     return left_weight, right_weight


# blend two images based on weights
# def blend_image(left_image, right_image, left_weight, right_weight):
#     left_img = cv2.imread(left_image)
#
#     right_img = cv2.imread(right_image)
#
#     blended = cv2.addWeighted(left_img, left_weight, right_img, right_weight, 0)
#
#     return blended


# detect dark image and replac them with liner interpolated image
# def check_discard_merge(options: List[ImageInput], replace: bool = False, threshold: float = 0.1):
#     # create and assign index list for dark image
#     idx_dark_imglist = [0] * len(options)
#
#     result_list = []
#
#     for idx, image in enumerate(options):
#         img_name, mean_luminosity, luminosity_str = isbright(image.input_file,
#                                                              threshold)  # luminosity detection, luminosity_str is either 'dark' or 'bright'
#         result_list.append([img_name, mean_luminosity, luminosity_str])
#         idx_dark_imglist[idx] = -1 if luminosity_str == 'dark' else (idx)
#
#     table = tabulate(result_list, headers=['image_file_name', 'luminous_avg', 'dark_or_bright'], tablefmt='orgtbl')
#     print(table + "\n")
#
#     # save dark image detection result as excel file
#     write_results_to_excel(result_list, options[0].output_directory)
#
#     # print(idx_dark_imglist)
#
#     # Finding consecutive occurrences of -1 in an array
#     max_dark_list_length = max(len(list(v)) for g, v in itertools.groupby(idx_dark_imglist))
#
#     # check max dark image sequence length, current method only deal with case with length equals 2
#     # print(max_dark_list_length)
#
#     # find dark image index
#     idx_dark = [i for i, x in enumerate(idx_dark_imglist) if x == -1]
#     idx_light = [i for i, x in enumerate(idx_dark_imglist) if x != -1]
#
#     # print(idx_dark)
#
#     # print(len(idx_dark_imglist))
#
#     # process dark image
#     if len(idx_dark) > 1:
#
#         for idx, value in enumerate(idx_dark):
#
#             # print("current value:{0}".format(value))
#
#             # if dark image appears as the start of image list
#             if value == 0:
#
#                 right_image_idx = ((value + 1) if idx_dark_imglist[value + 1] != -1 else (value + 2))
#
#                 left_image_idx = right_image_idx
#
#             # if dark image appears as the end of image list
#             elif value == len(idx_dark_imglist) - 1:
#
#                 left_image_idx = ((value - 1) if idx_dark_imglist[value - 1] != -1 else (value - 2))
#
#                 right_image_idx = left_image_idx
#
#             else:
#
#                 left_image_idx = ((value - 1) if idx_dark_imglist[value - 1] != -1 else (value - 2))
#
#                 right_image_idx = ((value + 1) if idx_dark_imglist[value + 1] != -1 else (value + 2))
#
#             # print("current image idx:{0}, left_idx:{1}, right_idx:{2}\n".format(value, left_image_idx, right_image_idx))
#
#             (left_weight, right_weight) = blend_weight_calculator(left_image_idx, right_image_idx, value)
#
#             # print("current image idx:{0}, left_idx:{1}, right_idx:{2}, left_weight:{3}, right_weight:{4} \n".format(value, left_image_idx, right_image_idx, left_weight, right_weight))
#
#             blended = blend_image(options[left_image_idx].input_file, options[right_image_idx].input_file, left_weight, right_weight)
#
#             print("Blending image:{0}, left:{1}, right:{2}, left_weight:{3:.2f}, right_weight:{4:.2f}".format(options[value].input_stem,
#                                                                                                               options[left_image_idx].input_stem,
#                                                                                                               options[right_image_idx].input_stem,
#                                                                                                               left_weight, right_weight))
#
#             # save result by overwriting original files
#             # cv2.imwrite(options[value].input_file, blended)
#
#             # save result into result folder for debugging
#             cv2.imwrite(join(options[0].output_directory,
#                              options[value].input_file) if replace else f"{join(options[0].output_directory, options[value].input_stem)}.blended.png",
#                         blended)
#
#     for idx, value in enumerate(idx_light):
#         image = cv2.imread(options[value].input_file)
#         cv2.imwrite(join(options[0].output_directory,
#                          options[value].input_file) if replace else f"{join(options[0].output_directory, options[value].input_stem)}.png", image)


# def check_discard_merge2(options: List[ImageInput], threshold: float = 0.1):
#     left = None
#     right = None
#     i = 0
#     replaced = 0
#     any_dark = False
#
#     # if every image has timestamp data, sort images by timestamp
#     if all(option.timestamp is not None for option in options):
#         options = sorted(options, key=lambda o: o.timestamp)
#
#     for option in options:
#         img_name, mean_luminosity, luminosity_str = isbright(option, threshold)  # luminosity detection, luminosity_str is either 'dark' or 'bright'
#         write_results_to_csv([(img_name, mean_luminosity, luminosity_str)], option.output_directory)
#         if luminosity_str == 'dark':
#             print(f"{option.input_stem} is too dark, skipping")
#             any_dark = True
#             continue
#         else:
#             path = join(options[0].output_directory, Path(option.input_file).name)
#             print(f"Writing to {path}")
#             cv2.imwrite(path, cv2.imread(option.input_file))
#         if luminosity_str == 'dark':
#             if left is None:
#                 left = i
#             right = i
#             any_dark = True
#         elif left is not None and left != right:
#             ii = 0
#             left_file = sorted_options[left].input_file
#             right_file = sorted_options[right].input_file
#             prev_file = sorted_options[left - 1].input_file
#             next_file = sorted_options[right + 1].input_file
#             prev = cv2.imread(prev_file)
#             next = cv2.imread(next_file)
#             width = right - left + 1
#             offset = (1 / width)
#             print(f"Replacing {left_file} to {right_file} with merger of {prev_file} and {next_file}")
#             for opt in reversed(sorted_options[left:right + 1]):
#                 prev_weight = ((ii / (right - left)) * ((width - 1) / width)) + offset
#                 next_weight = ((1 - prev_weight) * ((width - 1) / width)) + offset
#                 print(f"Merging {prev_file} (weight {round(prev_weight, 2)}) with {next_file} (weight: {round(next_weight, 2)})")
#                 blended = cv2.addWeighted(prev, prev_weight, next, next_weight, 0)
#                 # cv2.imwrite(opt.input_file, blended)
#                 cv2.imwrite(join(options[0].output_directory, opt.input_file) if replace else f"{join(options[0].output_directory, opt.input_stem)}.blended.png", blended)
#                 ii += 1
#             left = None
#             right = None
#             replaced += width
#         else:
#             cv2.imwrite(join(options[0].output_directory, option.input_file) if replace else f"{join(options[0].output_directory, option.input_stem)}.png", cv2.imread(option.input_file))
#         i += 1
#     print(f"Replaced {replaced} dark images with weighted blends of adjacent images")
#     return any_dark


# if every image has timestamp data, sort images by timestamp
#     if all(option.timestamp is not None for option in inputs):
#         inputs = sorted(inputs, key=lambda o: o.timestamp)



# def check_discard_merge(options: List[InputImage]):
#     left = None
#     right = None
#     i = 0
#     replaced = 0
#     any_dark = False
#     sorted_options = sorted(options, key=lambda o: o.timestamp)
#     for option in sorted_options:
#         img_name, mean_luminosity, luminosity_str = check_luminosity(option)  # luminosity detection, luminosity_str is either 'dark' or 'bright'
#         write_luminosity_results_to_csv([(img_name, mean_luminosity, luminosity_str)], option.output_directory)
#         if luminosity_str == 'dark':
#             if left is None:
#                 left = i
#             right = i
#             any_dark = True
#         elif left is not None and left != right:
#             ii = 0
#             left_file = sorted_options[left].path
#             right_file = sorted_options[right].path
#             prev_file = sorted_options[left - 1].path
#             next_file = sorted_options[right + 1].path
#             prev = cv2.imread(prev_file)
#             next = cv2.imread(next_file)
#             width = right - left + 1
#             offset = (1 / width)
#             print(f"Replacing {left_file} to {right_file} with merger of {prev_file} and {next_file}")
#             for opt in reversed(sorted_options[left:right + 1]):
#                 prev_weight = ((ii / (right - left)) * ((width - 1) / width)) + offset
#                 next_weight = ((1 - prev_weight) * ((width - 1) / width)) + offset
#                 print(f"Merging {prev_file} (weight {round(prev_weight, 2)}) with {next_file} (weight: {round(next_weight, 2)})")
#                 blended = cv2.addWeighted(prev, prev_weight, next, next_weight, 0)
#                 cv2.imwrite(opt.path, blended)
#                 cv2.imwrite(join(opt.output_directory, f"{opt.stem}.blended.png"), blended)
#                 ii += 1
#             left = None
#             right = None
#             replaced += width
#         i += 1
#     print(f"Replaced {replaced} dark images with weighted blends of adjacent images")
#     return any_dark