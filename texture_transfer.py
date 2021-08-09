from argparse import ArgumentParser
from PIL import Image   
import numpy as np
from scipy import signal
import cv2
from PIL import ImageFilter

#return a luminance_diff.shape[0] array indicate which col is turning point
def graph_cut_vertical(luminance_diff):
    ret_array = np.zeros((luminance_diff.shape[0]))
    dp_val_table = np.zeros(luminance_diff.shape)
    dp_dir_table = np.zeros(luminance_diff.shape)

    for i in range(luminance_diff.shape[1]):
        dp_val_table[0][i] = luminance_diff[0][i]
    
    #dp find min-cut
    for row in range(1, luminance_diff.shape[0], 1):
        for col in range(luminance_diff.shape[1]):
            val_list = [dp_val_table[row - 1][col]]
            dir_list = [col]
            if(col - 1 >= 0):
                val_list.append(dp_val_table[row - 1][col-1])
                dir_list.append(col - 1)
            if(col + 1 < luminance_diff.shape[1]):
                val_list.append(dp_val_table[row - 1][col+1])
                dir_list.append(col + 1)

            which = np.array(val_list).argmin()
            dp_val_table[row][col] = val_list[which] + luminance_diff[row][col]
            dp_dir_table[row][col] = dir_list[which]
    
    #print(dp_val_table)
    #print(dp_dir_table)
    last_row = luminance_diff.shape[0] - 1
    which_min = 0
    min_val = dp_val_table[last_row][0]
    for i in range(luminance_diff.shape[1]):
        if(min_val > dp_val_table[last_row][i]):
            which_min = i
            min_val = dp_val_table[last_row][i]
    
    #backtracking
    next_col = which_min
    for row in range(last_row, -1, -1):
        ret_array[row] = next_col
        next_col = int(dp_dir_table[row][next_col])
    
    return ret_array

#return a luminance_diff.shape[1] array indicate which row is turning point
def graph_cut_horizontal(luminance_diff):
    ret_array = np.zeros((luminance_diff.shape[1]))
    dp_val_table = np.zeros(luminance_diff.shape)
    dp_dir_table = np.zeros(luminance_diff.shape)

    for i in range(luminance_diff.shape[0]):
        dp_val_table[i][0] = luminance_diff[i][0]
    
    #dp find min-cut
    for col in range(1, luminance_diff.shape[1], 1):
        for row in range(luminance_diff.shape[0]):
            val_list = [dp_val_table[row][col - 1]]
            dir_list = [row]
            if(row - 1 >= 0):
                val_list.append(dp_val_table[row - 1][col - 1])
                dir_list.append(row - 1)
            if(row + 1 < luminance_diff.shape[0]):
                val_list.append(dp_val_table[row + 1][col - 1])
                dir_list.append(row + 1)

            which = np.array(val_list).argmin()
            dp_val_table[row][col] = val_list[which] + luminance_diff[row][col]
            dp_dir_table[row][col] = dir_list[which]
    
    #print(dp_val_table)
    #print(dp_dir_table)
    last_col = luminance_diff.shape[1] - 1
    which_min = 0
    min_val = dp_val_table[0][last_col]
    for i in range(luminance_diff.shape[0]):
        if(min_val > dp_val_table[i][last_col]):
            which_min = i
            min_val = dp_val_table[i][last_col]
    
    #backtracking
    next_row = which_min
    for col in range(last_col, -1, -1):
        ret_array[col] = next_row
        next_row = int(dp_dir_table[next_row][col])
    
    return ret_array

def quilt_patch_horizontal(patch_left, patch_right, overlap_len):
    patch_left_row, patch_left_col, _ = patch_left.shape
    patch_right_row, patch_right_col, _ = patch_right.shape

    if(patch_left_row != patch_right_row):
        print("patch_left_row need equal to patch_right_row")
        return -1
    if(overlap_len == 0):
        return np.concatenate((patch_left, patch_right), axis = 1)
    if(patch_right_col < overlap_len or patch_left_col < overlap_len):
        print("patch is smaller than overlap_len")
        return -1

    overlap_left = patch_left[0: patch_left_row, patch_left_col - overlap_len : patch_left_col, : ]
    overlap_right = patch_right[0: patch_right_row, 0 : overlap_len, : ]
    luminance_diff = np.square(overlap_left[:,:,0] - overlap_right[:,:,0])
    turning_point = graph_cut_vertical(luminance_diff)

    overlap_area = []
    for row in range(patch_left_row):

        now_row_left_data = patch_left[row : row + 1, patch_left_col - overlap_len: patch_left_col - overlap_len + int(turning_point[row]), :]
        now_row_right_data = patch_right[row : row + 1, int(turning_point[row]): overlap_len, :]
        overlap_area.append(np.concatenate((now_row_left_data, now_row_right_data) ,  axis = 1 ))
    
    overlap_area = np.concatenate(overlap_area, axis = 0)
    patch_left_remain = patch_left[0: patch_left_row, 0: patch_left_col - overlap_len, :]
    patch_right_remain = patch_right[0: patch_right_row, overlap_len: patch_right_col, :]
    #print(overlap_area.shape)
    #print(patch_left_remain.shape)
    #print(patch_right_remain.shape)
    return np.concatenate((patch_left_remain, overlap_area, patch_right_remain), axis = 1)


def quilt_patch_vertical(patch_top, patch_bottom, overlap_len):
    patch_top_row, patch_top_col, _ = patch_top.shape
    patch_bottom_row, patch_bottom_col, _ = patch_bottom.shape

    if(patch_top_col != patch_bottom_col):
        print("patch_top_col need equal to patch_bottom_col")
        return -1
    if(overlap_len == 0):
        return np.concatenate((patch_top,  patch_bottom), axis = 0)
    if(patch_bottom_row < overlap_len or patch_top_row < overlap_len):
        print("patch is smaller than overlap_len")
        return -1

    overlap_top = patch_top[patch_top_row - overlap_len : patch_top_row, 0 : patch_top_col, : ]
    overlap_bottom = patch_bottom[0 : overlap_len, 0 : patch_bottom_col, : ]
    luminance_diff = np.square(overlap_top[:,:,0] - overlap_bottom[:,:,0])
    turning_point = graph_cut_horizontal(luminance_diff)

    overlap_area = []
    for col in range(patch_top_col): 

        now_row_top_data = patch_top[patch_top_row - overlap_len : patch_top_row - overlap_len +  int(turning_point[col]), col : col + 1, : ]
        now_row_bottom_data = patch_bottom[int(turning_point[col]): overlap_len, col : col + 1, : ]
        overlap_area.append(np.concatenate((now_row_top_data, now_row_bottom_data) ,  axis = 0 ))
    
    overlap_area = np.concatenate(overlap_area, axis = 1)
    patch_top_remain = patch_top[0 : patch_top_row - overlap_len, 0 : patch_top_col, : ]
    patch_bottom_remain = patch_bottom[overlap_len : patch_bottom_row, 0 : patch_bottom_col, : ]
    #print(overlap_area)
    #print(patch_left_remain)
    #print(patch_right_remain)
    return np.concatenate((patch_top_remain, overlap_area, patch_bottom_remain), axis = 0)


def find_similar_block(texture_blur, block_blur, random):
    '''
    block_row, block_col = block_blur.shape
    texture_row, texture_col = texture_blur.shape

    min_val = np.absolute(texture_blur[0: block_row, 0: block_col] - block_blur).sum()
    min_row, min_col = 0, 0

    for row in range(0, texture_row - block_row):
        for col in range(0, texture_col - block_col):
            val = np.absolute(texture_blur[row: row + block_row, col: col + block_col] - block_blur).sum()
            if(min_val > val):
                min_val = val
                min_row, min_col = row, col
    return min_row, min_col
    '''
    res = cv2.matchTemplate(texture_blur.astype(np.float32),block_blur.astype(np.float32),cv2.TM_SQDIFF)
    if(random != 0):
        do_time = (np.random.randint(5))*(res.size)//random
        if(do_time >= res.size):
            do_time = res.size
        ind = np.unravel_index(np.argpartition(res, do_time, axis=None)[do_time], res.shape)
    else:
        ind = np.unravel_index(np.argmin(res,axis=None), res.shape)
    return ind

def texture_transfer(texture_data, target_pic_data, block_size, overlap_len, random):
    texture_luminance_blur = Image.fromarray(texture_data[:,:,0], 'L').filter(ImageFilter.GaussianBlur(radius=3))
    texture_luminance_blur = np.array(texture_luminance_blur)
    target_pic_luminance_blur = Image.fromarray(target_pic_data[:,:,0], 'L').filter(ImageFilter.GaussianBlur(radius=3))
    target_pic_luminance_blur = np.array(target_pic_luminance_blur)


    result_luminance = np.zeros((1, 1))
    result_rgb = np.zeros((1,1,1))
    for row in range(0, target_pic_data.shape[0], block_size - overlap_len):
       
        now_row_size = target_pic_data.shape[0] - row
        if(now_row_size > block_size):
            now_row_size = block_size
        if(now_row_size <= overlap_len):
            break

        patch_row, patch_col = find_similar_block(texture_luminance_blur, target_pic_luminance_blur[row: row + now_row_size, 0: block_size], random)
        now_row_data = texture_data[patch_row : patch_row + now_row_size, patch_col: patch_col + block_size]
        for col in range(block_size - overlap_len, target_pic_data.shape[1], block_size - overlap_len):
            #print((row, col))
            now_col_size = target_pic_data.shape[1] - col
            if(now_col_size > block_size):
                now_col_size = block_size
            if(now_col_size <= overlap_len):
                break
            patch_row, patch_col = find_similar_block(texture_luminance_blur, target_pic_luminance_blur[row : row + now_row_size, col: col + now_col_size], random)
            now_row_data = quilt_patch_horizontal(now_row_data, texture_data[patch_row: patch_row + now_row_size, patch_col: patch_col + now_col_size], overlap_len)
        if(row == 0):
            result_luminance = now_row_data
        else:
            result_luminance = quilt_patch_vertical(result_luminance, now_row_data, overlap_len)
        
    
    return result_luminance




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--text', '-t', type=str, dest="texture_path", required=True, help='path of texture image')
    parser.add_argument('--picture', '-p', type=str, dest="pic_path", required=True, help='path of source image')
    parser.add_argument('--outname', '-o', type=str, dest="outname", required=True, help='name of the output image')
    parser.add_argument('--overlap_len', type=int,default=10, help='the size of the overlap region between the patches')
    parser.add_argument('--patchsize', type=int, default=30, help='size of the patches')
    parser.add_argument('--random', type=int, default=2, help='random choose patch in some similar patches, 0~9.\n\
    This value becomes bigger, then #candicates of patches becomes larger.\n\
    0 is choose the most similar patch. That is #candicates of patches is one.')

    args = parser.parse_args()
    texture = Image.open(args.texture_path)
    texture = np.array(texture.convert('YCbCr'))
    target_pic = Image.open(args.pic_path)
    target_pic = np.array(target_pic.convert('YCbCr'))

    if(args.random!=0):
        args.random = (10-args.random)*100
    

    img_out = texture_transfer(texture, target_pic, args.patchsize, args.overlap_len, args.random)
    img_out = Image.fromarray(img_out, "YCbCr")
    img_out = img_out.convert("RGB")
    img_out.save(args.outname)


