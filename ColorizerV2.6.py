import numpy as np
import cv2
import PySimpleGUI as sg
import os.path
from PIL import ImageEnhance
from PIL import Image

version = 'Capstone 2023'

#Available themes
available_themes = sg.ListOfLookAndFeelValues()

prototxt = r'model/colorization_deploy_v2.prototxt'
model = r'model/colorization_release_v2.caffemodel'
points = r'model/pts_in_hull.npy'
points = os.path.join(os.path.dirname(__file__), points)
prototxt = os.path.join(os.path.dirname(__file__), prototxt)
model = os.path.join(os.path.dirname(__file__), model)

if not os.path.isfile(model):
    sg.popup_scrolled('Missing Data Model')
    exit()
net = cv2.dnn.readNetFromCaffe(prototxt, model)     # load model from disk
pts = np.load(points)

# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

#Parameter Functions
def colorize_image(image_filename=None, cv2_frame=None):
    # load the input image from disk, scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    image = cv2.imread(image_filename) if image_filename else cv2_frame
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # resize the Lab image to 224x224 (the dimensions the colorization network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    'print("[INFO] colorizing image...")'
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # grab the 'L' channel from the *original* input image (not the resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # the current colorized image is represented as a floating point data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    # print(image)
    return image, colorized

# Enhancements
def convert_to_grayscale(image):
    # Convert webcam frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert grayscale frame (single channel) to 3 channels
    gray_3_channels = np.zeros_like(image)
    gray_3_channels[:, :, 0] = gray
    gray_3_channels[:, :, 1] = gray
    gray_3_channels[:, :, 2] = gray
    return gray_3_channels

def adjust_saturation(image, saturation_factor):
    # Convert BGR image to RGB (PIL uses RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert RGB image to PIL Image
    pil_image = Image.fromarray(image_rgb)
    # Create a color enhancer
    enhancer = ImageEnhance.Color(pil_image)
    # Adjust the saturation of the image
    enhanced_image = enhancer.enhance(saturation_factor)
    # Convert PIL Image back to RGB format
    enhanced_image_rgb = np.array(enhanced_image)
    # Convert RGB image back to BGR format
    enhanced_image_bgr = cv2.cvtColor(enhanced_image_rgb, cv2.COLOR_RGB2BGR)
    return enhanced_image_bgr

def adjust_brightness(image, brightness_factor):
    # Perform brightness adjustment using OpenCV convertScaleAbs function
    adjusted_image = cv2.convertScaleAbs(
        image, alpha=brightness_factor, beta=0)
    return adjusted_image

def adjust_contrast(image, contrast_factor):
    # Perform contrast adjustment using OpenCV convertScaleAbs function
    adjusted_image = cv2.convertScaleAbs(
        image, alpha=contrast_factor, beta=128 * (1 - contrast_factor))
    return adjusted_image

def adjust_sharpness(image, sharpness_factor):
    # Perform sharpness adjustment using a 3x3 kernel
    kernel = np.array([[-sharpness_factor, -sharpness_factor, -sharpness_factor],
                       [-sharpness_factor, 1 + 8 *
                           sharpness_factor, -sharpness_factor],
                       [-sharpness_factor, -sharpness_factor, -sharpness_factor]])
    adjusted_image = cv2.filter2D(image, -1, kernel)
    return adjusted_image

def adjust_hue(image, hue_factor):
    # Ensure the input image is in BGR format and uint8 data type
    if image.dtype != np.uint8 or image.shape[-1] != 3:
        raise ValueError(
            "Input image must be in BGR format and uint8 data type.")

    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert RGB image to PIL Image
    pil_image = Image.fromarray(rgb_image)

    # Create a color enhancer
    enhancer = ImageEnhance.Color(pil_image)

    # Adjust the hue of the image
    enhanced_image = enhancer.enhance(hue_factor)

    # Convert PIL Image back to RGB format
    enhanced_image_rgb = np.array(enhanced_image)

    # Convert RGB image back to BGR format
    enhanced_image_bgr = cv2.cvtColor(enhanced_image_rgb, cv2.COLOR_RGB2BGR)

    return enhanced_image_bgr


def update_noise_reduction(image, kernel_size, sigma_x=2.0):
    # Ensure kernel_size is an odd number
    # Use bitwise OR operation to make sure it's odd
    kernel_size = int(kernel_size) | 1

    # Normalize NR_value to range [0.0, 1.0]
    sigma_x_normalized = NR_value * 5.0  # Scale to range [0.0, 5.0]

    NR_image = cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigma_x_normalized)
    return NR_image


def adjust_red(image, red_factor):
    # Adjust the red channel of the image
    image[:, :, 2] = np.clip(image[:, :, 2] * red_factor, 0, 255)
    return image

def adjust_green(image, green_factor):
    # Adjust the green channel of the image
    image[:, :, 1] = np.clip(image[:, :, 1] * green_factor, 0, 255)
    return image

def adjust_blue(image, blue_factor):
    # Adjust the blue channel of the image
    image[:, :, 0] = np.clip(image[:, :, 0] * blue_factor, 0, 255)
    return image

def adjust_Shadows_HL(image, highlight_factor=1.0, shadow_factor=1.0):
    # Adjust the contrast of the image with separate factors for highlights and shadows

    # Convert BGR image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Separate the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Normalize the L channel to the range [0, 1]
    l_normalized = l_channel / 255.0

    # Define the piecewise function to control the adjustment based on pixel intensity
    def adjust_pixel(x, x_low, x_high, factor_low, factor_high):
        return np.piecewise(x, [x <= x_low, x >= x_high], [lambda x: x * factor_low, lambda x: x * factor_high])

    # Adjust the L channel using the piecewise function for highlights and shadows separately
    l_adjusted = adjust_pixel(l_normalized, 0.5 - (1 - shadow_factor) / 2, 0.5, shadow_factor, 1)
    l_adjusted = adjust_pixel(l_adjusted, 0.5, 0.5 + (highlight_factor - 1) / 2, 1, highlight_factor)

    # Scale the L channel back to the range [0, 255]
    l_adjusted = (l_adjusted * 255).astype(np.uint8)

    # Merge the adjusted L channel with the original A and B channels
    lab_adjusted = cv2.merge((l_adjusted, a_channel, b_channel))

    # Convert the adjusted LAB image back to BGR color space
    bgr_adjusted = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2BGR)

    return bgr_adjusted
# --------------------------------- The GUI ---------------------------------

def create_layout():
    left_col = [
        [sg.Text('Folder'), sg.In(size=(25, 1), enable_events=True,
                                  key='-FOLDER-'), sg.FolderBrowse()],
        [sg.Listbox(values=[], enable_events=True,
                    size=(40, 10), key='-FILE LIST-')],
        [sg.CBox('Convert to gray first', key='-MAKEGRAY-')],
        [sg.Text('Term Project: ' + version, font='Courier 8')]
    ]

    images_col = [
        [sg.Text('Input file:'), sg.In(enable_events=True,
                                       key='-IN FILE-'), sg.FileBrowse()],
        [sg.Button('Colorize Photo', key='-PHOTO-'),
         sg.Button('Save File', key='-SAVE-'), sg.Button('Exit')],
        [sg.Column([[sg.Image(filename='', key='-IN-')]], expand_x=True, expand_y=True, justification='center'),
         sg.Column([[sg.Image(filename='', key='-OUT-')]], expand_x=True, expand_y=True, justification='center')]
    ]

    right_col = [
        [sg.Button('Update', key='-Update-', enable_events=True),
         sg.Button('Reset', key='-Reset-', enable_events=True)],
        [sg.Text('Filters', font='Courier 8')],
        [sg.Slider(orientation='horizontal', key='-BRIGHTNESS-', range=(0.0, 2.0), resolution=0.01, default_value=1, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetBrightness-'), sg.Text('Brightness', font='Courier 14', justification='left')],
        [sg.Slider(orientation='horizontal', key='-CONTRAST-', range=(0.0, 2.0), resolution=0.01, default_value=1, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetContrast-'), sg.Text('Contrast', font='Courier 14', justification='left')],
        [sg.Slider(orientation='horizontal', key='-SHARPNESS-', range=(0.0, 1.0), resolution=0.01, default_value=0, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetSharpness-'), sg.Text('Sharpness', font='Courier 14', justification='left')],
        [sg.Slider(orientation='horizontal', key='-SATURATION-', range=(0, 2.0), resolution=0.01, default_value=1, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetSaturation-'), sg.Text('Saturation', font='Courier 14', justification='left')],
        [sg.Slider(orientation='horizontal', key='-HUE-', range=(-3.0, 3.0), resolution=0.01, default_value=0, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetHue-'), sg.Text('Hue', font='Courier 14', justification='left')],
        [sg.Slider(orientation='horizontal', key='-GAMMA-', range=(0.1, 5.0), resolution=0.1, default_value=1, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetGamma-'), sg.Text('Gamma', font='Courier 14', justification='left')],
        [sg.Slider(orientation='horizontal', key='-NOISEREDUCTION-', range=(0.0, 1.0), resolution=0.01, default_value=0, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetNR-'), sg.Text('Noise Reduction', font='Courier 14', justification='left')],
        [sg.Slider(orientation='horizontal', key='-RED-', range=(0.0, 2.0), resolution=0.01, default_value=1, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetRed-'), sg.Text('Red', font='Courier 14', text_color='red', justification='left')],
        [sg.Slider(orientation='horizontal', key='-GREEN-', range=(0.0, 2.0), resolution=0.01, default_value=1, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetGreen-'), sg.Text('Green', font='Courier 14', text_color='green', justification='left')],
        [sg.Slider(orientation='horizontal', key='-BLUE-', range=(0.0, 2.0), resolution=0.01, default_value=1, enable_events=True),
         sg.Button('R', size=(1, 1), key='-ResetBlue-'), sg.Text('Blue', font='Courier 14', text_color='blue', justification='left')],
        [sg.Slider(orientation='horizontal', key='-HIGHLIGHTS-', range=(0.0, 2.0), resolution=0.01, default_value=1,
                   enable_events=True), sg.Button('R', size=(1, 1), key='-ResetHighlights-'), sg.Text('Highlights', font='Courier 14')],
        [sg.Slider(orientation='horizontal', key='-SHADOWS-', range=(0.0, 2.0), resolution=0.01, default_value=1,
                   enable_events=True), sg.Button('R', size=(1, 1), key='-ResetShadows-'), sg.Text('Shadows', font='Courier 14')]
    ]

    menu_def = [['File', ['Open', 'Exit']],
                ['Edit', ['Reset Filters']],
                ['Theme', available_themes]]

    layout = [
        [sg.Menu(menu_def, key='-MENU-')],
        [sg.Column(left_col), sg.VSeperator(), sg.Column(
            images_col, expand_x=True, expand_y=True), sg.VSeperator(), sg.Column(right_col)]
    ]
    return layout

# Create the initial window with the default theme
selected_theme = sg.theme()
window = sg.Window('Photo Colorizer', create_layout(), finalize=True, grab_anywhere=True, resizable=True)

# Update the theme menu values after the window creation
menu_def = [['File', ['Open', 'Exit']],
            ['Edit', ['Reset Filters']],
            ['Theme', available_themes]]
window['-MENU-'].update(menu_def)

#Event Loop
prev_filename = colorized = cap = None
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    elif event in (None, 'Open'):
        # Open file browser to directory of colorized photos
        break
    elif event in available_themes:
        selected_theme = event
        window.close()  # Close the current window
        sg.theme(selected_theme)
        # Create a new window with the selected theme
        window = sg.Window('Photo Colorizer', create_layout(
        ), finalize=True, grab_anywhere=True, resizable=True)
        # Update the theme menu values after the window creation
        menu_def = [['File', ['Open', 'Exit']],
                    ['Edit', ['Reset Filters']],
                    ['Theme', available_themes]]
        window['-MENU-'].update(menu_def)
        sg.popup_quick_message(
            "Theme selected: {selected_theme}", auto_close_duration=2, non_blocking=True)
        
    elif event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER-']
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
        # get list of files in folder
        try:
            flist0 = os.listdir(folder)
        except:
            continue
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
        window['-FILE LIST-'].update(fnames)

    elif event == '-FILE LIST-':    # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values['-FOLDER-'], values['-FILE LIST-'][0])
            image = cv2.imread(filename)
            window['-IN-'].update(data=cv2.imencode('.png',
                                  image)[1].tobytes())
            window['-OUT-'].update(data='')
            window['-IN FILE-'].update('')

            if values['-MAKEGRAY-']:
                gray_3_channels = convert_to_grayscale(image)
                window['-IN-'].update(data=cv2.imencode('.png',
                                      gray_3_channels)[1].tobytes())
                image, colorized = colorize_image(cv2_frame=gray_3_channels)
            else:
                image, colorized = colorize_image(filename)

            window['-OUT-'].update(data=cv2.imencode('.png',
                                   colorized)[1].tobytes())
        except:
            continue

    elif event == '-PHOTO-':        # Colorize photo button clicked
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(
                    values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue
            if values['-MAKEGRAY-']:
                gray_3_channels = convert_to_grayscale(cv2.imread(filename))
                window['-IN-'].update(data=cv2.imencode('.png',
                                      gray_3_channels)[1].tobytes())
                image, colorized = colorize_image(cv2_frame=gray_3_channels)
            else:
                image, colorized = colorize_image(filename)
                window['-IN-'].update(data=cv2.imencode('.png',
                                      image)[1].tobytes())
                window['-OUT-'].update(data=cv2.imencode('.png',
                                       colorized)[1].tobytes())
                # Temp comment-out to test brightness method!^^^
        except:
            continue

    #Single filename
    elif event == '-IN FILE-':      
        filename = values['-IN FILE-']
        if filename != prev_filename:
            prev_filename = filename
            try:
                image = cv2.imread(filename)
                window['-IN-'].update(data=cv2.imencode('.png',
                                      image)[1].tobytes())
            except:
                continue

    #Apply Values of Sliders to Photo
    elif event in ('-BRIGHTNESS-', '-CONTRAST-', '-SHARPNESS-', '-SATURATION-', '-HUE-', '-GAMMA-', '-Update-', '-NOISEREDUCTION-', '-RED-', '-GREEN-', '-BLUE-', '-HIGHLIGHTS-', '-SHADOWS-'):
        try:
            if values['-IN FILE-']:
                filename = values['-IN FILE-']
            elif values['-FILE LIST-']:
                filename = os.path.join(
                    values['-FOLDER-'], values['-FILE LIST-'][0])
            else:
                continue

            if values['-MAKEGRAY-']:
                gray_3_channels = convert_to_grayscale(cv2.imread(filename))
                _, colorized_image = colorize_image(cv2_frame=gray_3_channels)
            else:
                _, colorized_image = colorize_image(filename)

            # Apply filters based on slider values
            brightness_value = values['-BRIGHTNESS-']
            contrast_value = values['-CONTRAST-']
            sharpness_value = values['-SHARPNESS-']
            saturation_value = values['-SATURATION-']
            hue_value = values['-HUE-']
            gamma_value = values['-GAMMA-']
            NR_value = values['-NOISEREDUCTION-']
            red_value = values['-RED-']
            green_value = values['-GREEN-']
            blue_value = values['-BLUE-']
            highlight_value = values['-HIGHLIGHTS-']
            shadow_value = values['-SHADOWS-']

            #Call Adjustment Functions per Slider
            if brightness_value != 1.0:
                colorized_image = adjust_brightness(
                    colorized_image, brightness_value)

            if contrast_value != 1.0:
                colorized_image = adjust_contrast(
                    colorized_image, contrast_value)

            if sharpness_value != 0.0:
                colorized_image = adjust_sharpness(
                    colorized_image, sharpness_value)

            if saturation_value != 1.0:
                colorized_image = adjust_saturation(
                    colorized_image, saturation_value)

            if hue_value != 0.0:
                colorized_image = adjust_hue(colorized_image, hue_value)

            if gamma_value != 1.0:
                colorized_image = cv2.pow(colorized_image / 255.0, gamma_value)
                colorized_image = (255 * colorized_image).astype(np.uint8)

            if NR_value != 0.0:
                kernel_size = int(NR_value * 20) + 1
                colorized_image = update_noise_reduction(
                    colorized_image, kernel_size)

            if red_value != 1.0:
                colorized_image = adjust_red(colorized_image, red_value)

            if green_value != 1.0:
                colorized_image = adjust_green(colorized_image, green_value)

            if blue_value != 1.0:
                colorized_image = adjust_blue(colorized_image, blue_value)

            if highlight_value != 1.0 or shadow_value != 1.0:
                colorized_image = adjust_Shadows_HL(colorized_image, highlight_value, shadow_value)

            # Update the output image
            window['-OUT-'].update(data=cv2.imencode('.png',
                                   colorized_image)[1].tobytes())
        except Exception as e:
            print(e)

    #Reset Events
    elif event == '-Reset-':
        # Reset all sliders to default values
        window['-BRIGHTNESS-'].update(1.0)
        window['-CONTRAST-'].update(1.0)
        window['-SHARPNESS-'].update(0.0)
        window['-SATURATION-'].update(1.0)
        window['-HUE-'].update(0.0)
        window['-GAMMA-'].update(1.0)
        window['-NOISEREDUCTION-'].update(0.0)
        window['-RED-'].update(1.0)
        window['-GREEN-'].update(1.0)
        window['-BLUE-'].update(1.0)
        window['-HIGHLIGHTS-'].update(1.0)
        window['-SHADOWS-'].update(1.0)

        # Manually trigger the Update button event to apply the default adjustments
        window['-Update-'].click()
    # Individual Slider reset
    elif event == '-ResetBrightness-':
        # Reset slider to default value
        window['-BRIGHTNESS-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetContrast-':
        # Reset slider to default value
        window['-CONTRAST-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetSharpness-':
        # Reset slider to default value
        window['-SHARPNESS-'].update(0.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetSaturation-':
        # Reset slider to default value
        window['-SATURATION-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetHue-':
        # Reset slider to default value
        window['-HUE-'].update(0.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetGamma-':
        # Reset slider to default value
        window['-GAMMA-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetNR-':
        # Reset slider to default value
        window['-NOISEREDUCTION-'].update(0.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetRed-':
        # Reset slider to default value
        window['-RED-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetGreen-':
        # Reset slider to default value
        window['-GREEN-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetBlue-':
        # Reset slider to default value
        window['-BLUE-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetHighlights-':
        # Reset slider to default value
        window['-HIGHLIGHTS-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()
    elif event == '-ResetShadows-':
        # Reset slider to default value
        window['-SHADOWS-'].update(1.0)
        # Manually trigger the Update button event to apply the default adjustment
        window['-Update-'].click()

    #Save File buttom
    elif event == '-SAVE-' and colorized is not None:   
        filename = sg.popup_get_file(
            'Save colorized image.\nColorized image will be saved in a format matching the extension you enter.', save_as=True)
        try:
            if filename:
                cv2.imwrite(filename, colorized_image)
                sg.popup_quick_message(
                    'Image save complete', background_color='grey', text_color='white', font='Any 16')
        except:
            sg.popup_quick_message(
                'ERROR - Image NOT saved!', background_color='orange', text_color='white', font='Any 16')
# ----- Exit program -----
window.close()
