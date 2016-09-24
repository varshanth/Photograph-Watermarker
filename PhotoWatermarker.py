from PIL import Image
from sys import argv
import os
import numpy as np
import re
import debug_lib
import concurrent.futures
import itertools

debug_lib.GLOB_DEBUG_LEV = debug_lib.get_debug_lev('High')

_tolerance_perc = {
'window_pix':15,
'row_pix':10,
'col_pix':10,
'final_dev':12,
'sign_pix':5
}

_deviation_weights = {
'tot_window_dev':40,
'row_col_window_dev':60
}

_x_margin_pix_perc = 5
_y_margin_pix_perc = 3
_sign_composition = 1

def _get_title_from_docstring(fn_name):
    """
    Title: Get title from docstring
    Input: Function Name
    Output: Title from the Docstring of function
    """
    return 'Function: {0}'.format(
            debug_lib.get_match_from_docstring(fn_name,
                re.compile('Title: (.+)')))

def _dbg_fn_matrix_from_fn_title(fn_name):
    """
    Title: Debug Fn Matrix From Fn Title
    Input: Fn Name
    Output: {_get_title_from_docstring: [fn_name]}
    """
    return {_get_title_from_docstring:[fn_name]}

def _print_debug_msg_lev_high(fn_name, message):
    """
    Title: Wrapper for Print Level High Debug Message
    Input 1: Caller Function Name
    Input 2: Secondary Message
    Output: None
    """
    debug_lib.print_debug_msg_lev_high(_dbg_fn_matrix_from_fn_title(fn_name),
            message);

def _print_debug_msg_lev_medium(fn_name, message):
    """
    Title: Wrapper for Print Level Medium Debug Message
    Input 1: Caller Function Name
    Input 2: Secondary Message
    Output: None
    """
    debug_lib.print_debug_msg_lev_medium(_dbg_fn_matrix_from_fn_title(fn_name),
            message);

def _config_validator(config_file_path):
    """
    Title: Configuration Validator
    Input: Path to configuration File
    Output: None
    Functionality: Ensure validity of the config file
    """
    can_continue = True
    global _tolerance_perc
    global _deviation_weights
    global _x_margin_pix_perc
    global _y_margin_pix_perc
    global _sign_composition

    try:
        cfg_file = open(config_file_path,'r')
        cfg_content = cfg_file.read().split('\n')[:-1]
        cfgs = []
        for cfg in cfg_content:
            if re.match('^#', cfg) or re.match('^\s*$', cfg):
                continue
            cfg = re.findall('VALUE: ([1]?[0-9]?[0-9])', cfg)
            if len(cfg) != 1:
                raise TypeError('Configuration Value is wrong')
            cfgs.append(int(cfg[0]))
        if len(cfgs) < 10:
            raise RuntimeError('Number of Configurations is less than expected')
        if len(cfgs) > 10:
            raise RuntimeError('Number of Configurations is more than expected')

        # Copy configs
        _tolerance_perc['window_pix'] = cfgs[0]
        _tolerance_perc['row_pix'] = cfgs[1]
        _tolerance_perc['col_pix'] = cfgs[2]
        _tolerance_perc['final_dev'] = cfgs[3]
        _tolerance_perc['sign_pix'] = cfgs[4]
        _deviation_weights['tot_window_dev'] = cfgs[5]
        _deviation_weights['row_col_window_dev'] = cfgs[6]
        _x_margin_pix_perc = cfgs[7]
        _y_margin_pix_perc = cfgs[8]
        _sign_composition = cfgs[9]

        # Validate Specific Config Constraints
        if (_deviation_weights['tot_window_dev']
                + _deviation_weights['row_col_window_dev'] != 100):
            raise ValueError('Sum of Total Window Deviation Weight and Row and'\
                    ' Column Weight is not 100')

        if (_sign_composition > 1):
            raise ValueError('Signature Composition is not a valid value')

    except IOError as ioe:
        print 'Error in opening config file: {0}'.format(str(ioe))
        print 'Using default configs and proceeding'
    except Exception as e:
        print 'Error: {0}'.format(str(e))
        can_continue = False
    finally:
        cfg_file.close()
        if not can_continue:
            exit()

def _calculate_tolerance_range(average_val, tol_type):
    """
    Title: Calculate Tolerance Range
    Input 1: Average Value
    Input 2: Tolerance Type
    Functionality: Determine lower and upper bound based on Tolerance Type
    Output: [Lower Bound, UpperBound]
    """
    lower_bound = int(average_val * (1 - ((_tolerance_perc[tol_type])/100.0)))
    upper_bound = int(average_val * (1 + ((_tolerance_perc[tol_type])/100.0)))
    debug_msg = 'Input: Average Value = {0} Tolerance Type = {1}\n'.format(
            average_val, tol_type)
    debug_msg += 'Lower Bound {0}, Upper Bound {1}'.format(lower_bound,
            upper_bound)
    _print_debug_msg_lev_medium(_calculate_tolerance_range, debug_msg)
    return [lower_bound, upper_bound]

def _determine_signature_validity(signature_dim, photo_dim):
    """
    Title: Determine Signature Validity
    Input 1: Signature Image Dimensions
    Input 2: Photo Image Dimensions
    Output: True if Signature is smaller than photo, False otherwise
    """
    w_sign, h_sign = signature_dim
    w_photo, h_photo = photo_dim
    if w_photo < w_sign or h_photo < h_sign:
        return False
    return True

def _calc_cut_off_index(signature_window):
    """
    Title: Calculate Cut Off Index
    Input: Signature Pixel Matrix
    Output: Index of Row which crossed tolerated deviation
    """
    n_rows, n_cols = np.shape(signature_window)
    row_avg = [np.mean(row) for row in signature_window]
    for i in range(0, n_rows):
        tolerated_range = _calculate_tolerance_range(row_avg[i], 'sign_pix')
        dev_pix = filter(lambda pix:
                 (not(tolerated_range[0] <= pix <= tolerated_range[1])),
                 signature_window[i])
        if len(dev_pix) > 0:
            debug_msg = 'Pixel Deviation found at row {0}\n'.format(i)
            debug_msg += 'Number of Deviated Pixels is {0}'.format(len(dev_pix))
            _print_debug_msg_lev_high(_calc_cut_off_index, debug_msg)
            return i
    return 0

def _get_effective_sign_indices(s_pix):
    """
    Title: Get Effective Signature Indices
    Input: Signature Pixel Matrix (one channel only)
    Functionality: Return the effective signature indices
    Output: [Start Row Index, End Row Index, Start Col Index, End Col Index]
    """
    n_rows, n_cols = np.shape(s_pix)
    transposed_win = np.transpose(s_pix)
    upside_down_window = np.rot90(np.rot90(s_pix))
    upside_down_transposed_win = np.rot90(np.rot90(transposed_win))
    start_row_index = _calc_cut_off_index(s_pix)
    start_col_index = _calc_cut_off_index(transposed_win)
    end_row_index = n_rows -1 - _calc_cut_off_index(upside_down_window)
    end_col_index = n_cols -1 - _calc_cut_off_index(upside_down_transposed_win)
    rv = [start_row_index, end_row_index, start_col_index, end_col_index]
    debug_msg = ('[Start Row Idx, End Row Idx, Start Col Idx, End Col Idx] = '
            +str(rv))
    _print_debug_msg_lev_high(_get_effective_sign_indices, debug_msg)
    return rv

def get_effective_sign_indices_all_channels(s_pix_rgb):
    """
    Title: Get Effective Signature Indices From Channel-Wise Calculation
    Input: Signature Pixel Matrix containing RGB Channels
    Functionality: Return the channel-consolidated effective signature indices
    Output: [Start Row Index, End Row Index, Start Col Index, End Col Index]
    """
    #Go through each pixel in all the rows & cols and select the desired channel
    s_pix_r = s_pix_rgb[:,:,0]
    s_pix_g = s_pix_rgb[:,:,1]
    s_pix_b = s_pix_rgb[:,:,2]
    s_pix_rgb_list = [s_pix_r, s_pix_g, s_pix_b]
    channel_wise_eff_indices = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    with concurrent.futures.ProcessPoolExecutor() as ppexecutor:
        channel_wise_eff_indices = ppexecutor.map(_get_effective_sign_indices,
                s_pix_rgb_list)
    '''
    Rearrange so that to get 4 index sets
    [start row idx set, end row idx set, start col idx set, end row idx set]
    where index_set = (r_index, g_index, b_index)
    '''
    all_channels_indices_set = zip(*channel_wise_eff_indices)
    min_indices_of_channels = map(
            lambda all_channel_index_set: min(all_channel_index_set),
            all_channels_indices_set)
    max_indices_of_channels = map(
            lambda all_channel_index_set: max(all_channel_index_set),
            all_channels_indices_set)
    '''
    Start Index is chosen as the earliest encountered pixel which deviated above
        tolerance (when processed from LEFT to RIGHT)
    Start Index = Minimum(All Channels Start Indices)

    End Index is chosen as the earliest encountered pixel which deviated above
        tolerance (when processed from RIGHT to LEFT)
    End Index = Maximum(All Channels End Indices)
    '''
    effective_sign_indices = (
    #[Start Row Index, End Row Index, Start Col Index, End Col Index]
            [min_indices_of_channels[i] if i%2 == 0
                else max_indices_of_channels[i] for i in range(4)])

    debug_msg = 'All Channels Indices Set = {0}\n'.format(
            all_channels_indices_set)
    debug_msg += 'Effective Sign Indices = {0}'.format(
            effective_sign_indices)
    _print_debug_msg_lev_high(get_effective_sign_indices_all_channels,
            debug_msg)
    return effective_sign_indices

def get_effective_sign(sign_image):
    """
    Title: Get effective signature
    Input: Signature Image
    Output: Effective Signature Image
    """
    sign_RGB = sign_image.convert(matrix='RGB')
    s_pix_rgb = np.asarray(sign_RGB)
    start_row_idx, end_row_idx, start_col_idx, end_col_idx = \
            get_effective_sign_indices_all_channels(s_pix_rgb)
    effective_sign = Image.fromarray(
            s_pix_rgb[start_row_idx:end_row_idx+1,
                start_col_idx:end_col_idx+1, :], 'RGB')
    return effective_sign

def image_open(image_path):
    """
    Title: Open Image with Error Handling
    Input: Image Path
    Output: [Image File Descriptor, Pixel Matrix, Exception Object]
    """
    pix = 0
    try:
        local_image = Image.open(image_path)
        pix = local_image.load()
    except Exception as e:
        return [None, pix, e]
    return [local_image, pix, None]

def _calc_total_window_pix_deviation(window):
    """
    Title: Calculate Total Window Pixel Deviation
    Input: Pixel Matrix (window)
    Functionality: Calculate the window deviation percentage based on window avg
    Output: Total Window Deviation Percentage
    """
    window_avg = np.mean(window)
    tolerated_range = _calculate_tolerance_range(window_avg, 'window_pix')
    flat_window = [pixel for row in window for pixel in row]
    num_pix_flat_window = len(flat_window)
    deviated_pix = filter(lambda pix:
        (not(tolerated_range[0] <= pix <= tolerated_range[1])), flat_window)
    num_deviated_pix = len(deviated_pix)
    window_deviation_perc = num_deviated_pix * 100.0 / num_pix_flat_window
    debug_msg = 'Num Deviated Pix = {0}, Tot Pix = {1}\n'.format(
            num_deviated_pix, num_pix_flat_window)
    debug_msg += 'Total Windows Deviation = {0}'.format(window_deviation_perc)
    _print_debug_msg_lev_high(_calc_total_window_pix_deviation, debug_msg)
    return window_deviation_perc

def _calculate_row_wise_window_pix_deviation(window, tol_type = 'row_pix'):
    """
    Title: Calculate Row Wise Window Pixel Deviation
    Input 1: Pixel Matrix(window)
    Input 2: Tolerance Type (Default is Row Tolerance Value)
    Functionality: Calculate the window deviation percentage based on row avg
    Output: Row Wise Window Deviation Percentage
    """
    n_rows, n_cols = np.shape(window)
    row_avg = [np.mean(row) for row in window]
    num_deviations_per_row = [0 for i in range(n_rows)]
    for i in range(n_rows):
        tolerated_range = _calculate_tolerance_range(row_avg[i], tol_type)
        deviated_pix = filter(lambda pix:
            (not(tolerated_range[0] <= pix <= tolerated_range[1])), window[i])
        num_deviations_per_row[i] = len(deviated_pix)

    deviation_perc_per_row = map(
        lambda num_dev: (num_dev * 100.0/n_cols), num_deviations_per_row)
    avg_row_deviation_perc = np.mean(deviation_perc_per_row)
    debug_msg = 'Tolerance Type: {0}\n'.format(tol_type)
    debug_msg += 'Num Deviation Per Row: {0}\nNum of columns {1}\n'.format(
            num_deviations_per_row, n_cols)
    debug_msg += 'Average Row Wise Window Deviation = {0}'.format(
            avg_row_deviation_perc)
    _print_debug_msg_lev_high(_calculate_row_wise_window_pix_deviation,
            debug_msg)
    return avg_row_deviation_perc;

def _calculate_col_wise_window_pix_deviation(window):
    """
    Title: Calculate Column Wise Window Pixel Deviation
    Input: Pixel Matrix(window)
    Functionality: Calculate the window deviation percentage based on col avg
    Output: Column Wise Window Deviation Percentage
    """
    transpose = np.transpose(window)
    avg_col_deviation_perc = \
            _calculate_row_wise_window_pix_deviation(transpose, 'col_pix')
    return avg_col_deviation_perc


def _calc_window_deviation(p_pix):
    """
    Title: Calculate Final Window Pixel Deviation
    Input: Photo Pixel Matrix (One Channel Only)
    Functionality: Calculate the window deviation percentage based on a weighted
    average of total window deviation, row wise and column wise window deviation
    Output: Final Window Deviation Percentage
    """
    height, width = np.shape(p_pix)
    row_weight = _deviation_weights['row_col_window_dev']*(width/(width+height))
    col_weight = _deviation_weights['row_col_window_dev'] - row_weight
    tot_window_pix_deviation = _calc_total_window_pix_deviation(p_pix)
    row_wise_pix_deviation = _calculate_row_wise_window_pix_deviation(p_pix)
    col_wise_pix_deviation = _calculate_col_wise_window_pix_deviation(p_pix)
    final_deviation = (
            _deviation_weights['tot_window_dev'] * tot_window_pix_deviation
            + col_weight * col_wise_pix_deviation
            + row_weight * row_wise_pix_deviation)/100.0
    debug_msg = 'Final Deviation = {0}'.format(final_deviation)
    _print_debug_msg_lev_high(_calc_window_deviation, debug_msg)
    return final_deviation

def calc_window_deviation_all_channels(p_pix_rgb):
    """
    Title: Calculate Final Window Pixel Deviation for RGB channels
    Input: Photo Pixel Matrix with RGB Channels
    Functionality: Invoke Window Deviation calculation for RGB channels
    Output: Max(Deviation of R, G, B Individual Channels)
    """
    #Go through each pixel in all the rows & cols and select the desired channel
    p_pix_r = p_pix_rgb[:,:,0]
    p_pix_g = p_pix_rgb[:,:,1]
    p_pix_b = p_pix_rgb[:,:,2]
    p_pix_rgb_list = [p_pix_r, p_pix_g, p_pix_b]
    channel_dev = []
    for channel_mat in p_pix_rgb_list:
        channel_dev.append(_calc_window_deviation(channel_mat))

    max_dev = max(channel_dev)
    debug_msg = 'Channel Deviations = {0}, Max Deviation = {1}'.format(
            channel_dev, max_dev)
    _print_debug_msg_lev_high(calc_window_deviation_all_channels, debug_msg)
    return max_dev

def _quadrant_window_generator(w_photo, h_photo, w_sign, h_sign):
    """
    Title: Quadrant Window Generator
    Input 1: Width of photo
    Input 2: Height of photo
    Input 3: Width of Effective Signature
    Input 4: Height of Effective Signature
    Output: [Start Col Index, End Col Index, Start Row Index, End Row Index]
    for each quadrant
    """
    x_margin = int(_x_margin_pix_perc * w_photo / 100.0)
    y_margin = int(_y_margin_pix_perc * h_photo / 100.0)
    x_min_photo = x_margin
    x_max_photo = w_photo - x_margin - 1
    y_min_photo = y_margin
    y_max_photo = h_photo - y_margin - 1
    # X-Min and X-Max are calculated W.R.T the Cartesian Plane,
    # ------->          <-------
    # |                        |
    # |                        |
    # so, they correspond to the "Column Indices"
    # Y-Min and Y-Max are calculated W.R.T the Cartesian Plane,
    # |
    # v
    #
    #
    # ^
    # |
    # so, they correspond to the "Row Indices"

    #Coordinates for top left quadrant
    yield((x_min_photo, x_min_photo+w_sign, y_min_photo, y_min_photo+h_sign))
    #Coordinates for top right quadrant
    yield((x_max_photo-w_sign, x_max_photo, y_min_photo, y_min_photo+h_sign))
    #Coordinates for bottom left quadrant
    yield((x_min_photo, x_min_photo+w_sign, y_max_photo-h_sign, y_max_photo))
    #Coordinates for bottom right quadrant
    yield((x_max_photo-w_sign, x_max_photo, y_max_photo-h_sign, y_max_photo))

def _get_optimal_window_for_overlay(photo_RGB, w_sign, h_sign):
    """
    Title: Get Optimal Window for Overlay
    Input 1: Photo RGB Image
    Input 2: Width of Effective Sign
    Input 3: Height of Effective Sign
    Output: Quadrant Window with least acceptable deviation
    Functionality: Analyze the deviation for each quadrant and select the
    least deviated window for signature overlay
    """
    w_photo, h_photo = photo_RGB.size
    p_pix_rgb = np.asarray(photo_RGB)
    quadrant_dev = {}
    debug_msg = ''
    for quadrant in _quadrant_window_generator(w_photo, h_photo,
            w_sign, h_sign):
        debug_msg_q = 'Processing Quadrant {0}'.format(list(quadrant))
        _print_debug_msg_lev_high(_get_optimal_window_for_overlay, debug_msg_q)
        quadrant_dev[quadrant] = calc_window_deviation_all_channels(
            p_pix_rgb[quadrant[2]:quadrant[3], quadrant[0]:quadrant[1],:])
        debug_msg += 'Quadrant = {0}, Deviation = {1}\n'.format(
                list(quadrant), quadrant_dev[quadrant])

    quad_min_deviation = min(quadrant_dev, key=quadrant_dev.get)
    if quadrant_dev[quad_min_deviation] > _tolerance_perc['final_dev']:
        selected_quad = ()
    else:
        selected_quad = quad_min_deviation

    debug_msg += 'Selected Quadrant = {0}'.format(list(selected_quad))
    _print_debug_msg_lev_high(_get_optimal_window_for_overlay, debug_msg)
    return selected_quad

def overlay_signature_on_photo(photo_RGB, sign_RGB, _sign_composition):
    """
    Title: Overlay Signature on photo
    Input 1: Photo RGB Image
    Input 2: Effective Sign RGB Image
    Output: None but the original Photo RGB Image will be changed
    """
    selected_quadrant = _get_optimal_window_for_overlay(photo_RGB,
            *sign_RGB.size)
    if len(selected_quadrant) is 0:
        print ('Error: Photo cannot be watermarked given the configured'\
        ' tolerance levels')
        exit()

    j_min, j_max, i_min, i_max = selected_quadrant
    debug_msg = 'Signature Composition = {0}\n'.format(_sign_composition)
    for i,j in itertools.product(range(i_min,i_max), range(j_min,j_max)):
        # The "max" index returned by the quadrant list is calculated using
        # the "min" index + width/height of sign. This doesn't include the
        # min'th index itself; hence it is in "set" terms, [min,max) i.e,
        # max = max'th index + 1
        sign_i = i - i_min
        sign_j = j - j_min
        # Why (j, i)? Image.getpixel accepts Cartesian coordinates (x,y)
        sign_pix = sign_RGB.getpixel((sign_j, sign_i))
        # Determine if signature pixel is light or dark
        # A pixel is considered light, if 2 out of the 3 channels leans
        # towards 255 (since RGB = (255,255,255) is white)
        is_light_pix_sign = len(filter(lambda channel: channel > 127,
            sign_pix)) >= 2
        #If signature pixel and signature composition is same
        #then overlay signature to photo
        if ((is_light_pix_sign and _sign_composition != 1) or
                ((not is_light_pix_sign) and _sign_composition != 0)):
            photo_pix = photo_RGB.getpixel((j,i))
            is_light_pix_photo = len(filter(lambda channel: channel > 127,
                photo_pix)) >= 2
            debug_msg += 'Is Photo Pixel Light = {0}\n'.format(
                    is_light_pix_photo)
            debug_msg += 'Sign Pixel ({0},{1}) changes Photo Pixel ({2},{3})\n'\
                    .format(sign_j, sign_i, j, i)
            if is_light_pix_photo:
                photo_RGB.putpixel((j,i),(0,0,0))
            else:
                photo_RGB.putpixel((j,i),(255,255,255))
    _print_debug_msg_lev_medium(overlay_signature_on_photo, debug_msg)

def main():
    """
    Title: Main Program
    Input 1: Path to the photo
    Input 2: Path to the signature
    Input 3: Path to save the watermarked photo (including name)
    Input 4: Path to the config file
    Input 5: Percentage scale of signature to use for overlay
    """
    pass

if __name__ == '__main__':
    main()
    argc = len(argv)
    if (argc < 6):
        print '''
        1st ARG: Path to the Photo\n
        2nd Arg: Path to the Signature\n
        3rd ARG: Path to save the watermarked photo (including name)\n
        4th ARG: Path to the config file\n
        5th ARG: Percentage scale of signature to use for overlay
        '''
        exit()
    elif (argc > 6):
        print 'Too many args'
        exit()

    photo_path, sign_path = argv[1:3]
    watermarked_photo_path = argv[3]
    config_file_path = argv[4]
    sign_perc = argv[5]

    photo, p_pix, p_err = image_open(photo_path)
    sign, s_pix, s_err = image_open(sign_path)
    if photo is None:
        print 'Error in opening the photo: {0}'.format(str(p_err))
        exit()

    if sign is None:
        print 'Error in opening the sign: {0}'.format(str(s_err))
        exit()

    if (len(re.findall('^([1]?[0-9]?[0-9])$', sign_perc)) < 1):
        print 'Percentage scale of signature is incorrect'
        exit()

    _config_validator(config_file_path)
    sign_perc = int(sign_perc)
    photo_RGB = photo.convert(matrix='RGB')
    sign_RGB = get_effective_sign(sign)
    w_sign, h_sign = sign_RGB.size
    w_photo, h_photo = photo_RGB.size
    w_sign_new = int(w_sign * (sign_perc/100.0))
    h_sign_new = int(h_sign * (sign_perc/100.0))
    debug_msg = 'New Signature Dimensions = ({0},{1})'.format(
            w_sign_new, h_sign_new)
    _print_debug_msg_lev_high(main, debug_msg)
    sign_RGB.thumbnail((w_sign_new, h_sign_new), Image.ANTIALIAS)
    sign_valid = _determine_signature_validity(sign_RGB.size, photo_RGB.size)
    if (not sign_valid):
        print 'Size of the signature exceeds that of the image'
        exit()

    #Here we go!
    overlay_signature_on_photo(photo_RGB, sign_RGB, _sign_composition)
    photo_RGB.save(watermarked_photo_path)
    debug_msg = '{0} is watermarked as {1}'.format(
            photo_path, watermarked_photo_path)
    _print_debug_msg_lev_high(main, debug_msg)