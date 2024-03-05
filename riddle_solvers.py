# Add the necessary imports here
import pandas as pd
import torch
from utils import *
import skimage.io as io

# Show the figures / plots inside the notebook
from skimage import io, color
from skimage.color import rgb2gray,rgb2hsv
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
import numpy as np

from skimage.exposure import histogram
from matplotlib.pyplot import bar
import cv2
import tensorflow as tf
import pickle
import joblib
from datetime import datetime, timedelta


def rotate_left(lst, n):
    return lst[n:] + lst[:n]

def solve_cv_easy(test_case: tuple) -> list:

    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing a shredded image.
        - An integer representing the shred width in pixels.

    Returns:
    list: A list of integers representing the order of shreds. When combined in this order, it builds the whole image.
    """
    try:
            shreded = np.array(test_case[0])    
            print(shreded.shape)        
            list_of_indexes = list()
            if len(shreded.shape) == 3:
                x,y,z=shreded.shape
            else:
                x,y=shreded.shape
            shreded_list = np.split(shreded, y // test_case[1], axis=1)
            
            solution=list()
            solution.append(shreded_list[0])
            list_of_indexes.append(0)
            numOfMatches = 0
            shreded_list_copy = shreded_list.copy()
        
            for i in range(1,len(shreded_list_copy)):
                concatenated_image = np.concatenate(solution, axis=1)
                for j in range(len(shreded_list)):
                    temp=np.sum(solution[i-1][:,-1]==shreded_list[j][:,0])
                    if temp>numOfMatches:
                        numOfMatches=temp
                        index=j
                solution.append(shreded_list[index])
                list_of_indexes.append(index)
                numOfMatches = 0
            concatenated_image = np.concatenate(solution, axis=1)
            return list_of_indexes
    except Exception as e:
            print("An error occurred in cv_easy riddle solver with input = {test_case} and output = {list_of_indexes}, error = {e}")
            return []    



def solve_cv_medium(input: tuple) -> list:
    combined_image_array , patch_image_array = input
    inputImage = np.array(combined_image_array,dtype=np.uint8)
    patch = np.array(patch_image_array,dtype=np.uint8)
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A numpy array representing the RGB base image.
        - A numpy array representing the RGB patch image.

    Returns:
    list: A list representing the real image.
    """
    #//////////////////////////////////////////////#
    try:
        dict_patch = {}
        x_patch,y_patch=patch.shape[0:2]
        for i in range(0,x_patch):
                for j in range(0,y_patch):
                    color_tuple = tuple(patch[i,j][0:3])
                    if color_tuple in dict_patch.keys():
                        dict_patch[color_tuple] += 1
                    else:
                        dict_patch[color_tuple] = 1
        sorted_dict = sorted(dict_patch.items(), key=lambda x: x[1], reverse=True)

        dict_image = {}
        x_image,y_image=inputImage.shape[0:2]
        sumOFCOMMONINBAse=0
        sumOFCOMMONIN_patch=0
        for i in range(0,x_image):
                for j in range(0,y_image):
                    color_tuple = tuple(inputImage[i,j][0:3])
                    if color_tuple in dict_patch.keys():
                        sumOFCOMMONINBAse+=1
                        if color_tuple in dict_image.keys():
                            dict_image[color_tuple] += 1
                        else:
                            dict_image[color_tuple] = 1        
        sorted_dict_baseImage = sorted(dict_image.items(), key=lambda x: x[1], reverse=True)

            #create patch dic common colors and their frequency
        dict_patch_common = {}
        for i in sorted_dict:
                if i[0] in dict_image.keys():
                    dict_patch_common[i[0]] = i[1]
                    sumOFCOMMONIN_patch+=i[1]   

        sorted_dict_patch_common = sorted(dict_patch_common.items(), key=lambda x: x[1], reverse=True)

        ratio = np.sqrt(sumOFCOMMONINBAse)/patch.shape[0]
            #create patch dic common colors and their frequency
        dict_patch_common = {}
        for i in sorted_dict:
                if i[0] in dict_image.keys():
                    dict_patch_common[i[0]] = i[1]
                    sumOFCOMMONIN_patch+=i[1]   

        sorted_dict_patch_common = sorted(dict_patch_common.items(), key=lambda x: x[1], reverse=True)

        ratio = np.sqrt(sumOFCOMMONINBAse)/patch.shape[0]
        counter = 0
        x, y = inputImage.shape[0:2]
        target_color = np.array(sorted_dict[0][0][0:3])

        for i in range(x):
                for j in range(y):
                    if np.all(inputImage[i, j][0:3] == target_color):
                        counter += 1

        resizeX= int(patch.shape[0]*ratio)
        resizeY= int(patch.shape[1]*ratio)  
            


        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        input_gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

        patch_edges = cv2.Canny(patch_gray, 100, 250)
        input_edges = cv2.Canny(input_gray, 100, 250)

        best_match_value = -1
        best_match_scale = 1.0
        best_match_location = None

        if ratio<1:
                scaleValue=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        else:
                scaleValue=[ratio]
                
        for scale in scaleValue:
                resized_patch = cv2.resize(patch_edges, (0, 0), fx=scale, fy=scale)

                result = cv2.matchTemplate(input_edges, resized_patch, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if max_val > best_match_value:
                    best_match_value = max_val
                    best_match_scale = scale
                    best_match_location = max_loc

        top_left_x, top_left_y = best_match_location
        patch_height, patch_width = patch_gray.shape

        bottom_right_x, bottom_right_y = (
                int(top_left_x + patch_width * best_match_scale),
                int(top_left_y + patch_height * best_match_scale)
            )
        cv2.rectangle(inputImage, (top_left_x-7, top_left_y-7), (bottom_right_x+7, bottom_right_y+7), (0, 0, 0), -1)
            

            
        damaged_img = inputImage.copy()
        mask = np.zeros(damaged_img.shape[:2], dtype = "uint8")
        height, width = damaged_img.shape[0], damaged_img.shape[1]

        for i in range(height):
                for j in range(width):
                    if damaged_img[i, j].sum() > 0:
                        mask[i, j] = 0
                    else:
                        mask[i, j] = 255

            
        inputImage = inputImage[:,:,0:3]
        dst = cv2.inpaint(inputImage, mask, 3, cv2.INPAINT_NS)
        return dst.tolist()    
    except Exception as e:
        print(f"An error occurred in cv_medium riddle solver with input = {input} and output = {dst}, error = {e}")
        return []



    #//////////////////////////////////////////////#



def solve_cv_hard(input: tuple) -> int:
    extracted_question, image = test_case
    image = np.array(image)
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A string representing a question about an image.
        - An RGB image object loaded using the Pillow library.

    Returns:
    int: An integer representing the answer to the question about the image.
    """
    return 0


def solve_ml_easy(input: pd.DataFrame) -> list:
    data = pd.DataFrame(data)

    """
    This function takes a pandas DataFrame as input and returns a list as output.

    Parameters:
    input (pd.DataFrame): A pandas DataFrame representing the input data.

    Returns:
    list: A list of floats representing the output of the function.
    """
    try:
        loaded_model = joblib.load('my_model2.pkl')
        listofSol=[]
        forecasted_attacks = []
        current_date=data["timestamp"][0]

        for _ in range(50):
            current_date_formatted = current_date.strftime('%Y-%m-%d')
            current_data_point = np.array([[6]]) 
            current_data_point_reshaped = current_data_point.reshape((1, 1))  # Reshape to (1, 1) for SVR input
            next_data_point = loaded_model.predict(current_data_point_reshaped)
            forecasted_attacks.append(next_data_point[0])
            current_date += timedelta(days=1)
        return forecasted_attacks
    except Exception as e:
        print(f"An error occurred in ml_easy riddle solver with input = {input} and output =, error = {e}")
        return []


def solve_ml_medium(input: list) -> int:
    """
    This function takes a list as input and returns an integer as output.

    Parameters:
    input (list): A list of signed floats representing the input data.

    Returns:
    int: An integer representing the output of the function.
    """
    try:
        loaded_model = tf.keras.models.load_model(r"D:/dellHack/fox/HackTrick24-main/Solvers/riddle_ml_medium")
        input = np.array(input)

        input_lstm = input.reshape((1, 1, len(input)))
        prediction = loaded_model.predict(input_lstm)

        #print(prediction)
        rounded_prediction = np.round(prediction)
        predicted_class = int(rounded_prediction[0][0])  

        return  predicted_class
    except Exception as e:
        print(f"An error occurred in ml_medium riddle solver with input = {input} and output = {predicted_class}, error = {e}")
        return 0




def solve_sec_medium(input: torch.Tensor) -> str:
    img = torch.tensor(img)
    """
    This function takes a torch.Tensor as input and returns a string as output.

    Parameters:
    input (torch.Tensor): A torch.Tensor representing the image that has the encoded message.

    Returns:
    str: A string representing the decoded message from the image.
    """
    return ''

def solve_sec_hard(input:tuple)->str:
    """
    This function takes a tuple as input and returns a list a string.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A key 
        - A Plain text.

    Returns:
    list:A string of ciphered text
    """
    try:
    
        inputt=input[1]
        key=input[0]

        key_after_discarding=[] #56
        input_binary_permutated=[] #64

        convert_hex_bin={'0': "0000",
                        '1': "0001",
                        '2': "0010",
                        '3': "0011",
                        '4': "0100",
                        '5': "0101",
                        '6': "0110",
                        '7': "0111",
                        '8': "1000",
                        '9': "1001",
                        'A': "1010",
                        'B': "1011",
                        'C': "1100",
                        'D': "1101",
                        'E': "1110",
                        'F': "1111"}
        convert_hex_bin_1_15={'0': "0000",
                        '1': "0001",
                        '2': "0010",
                        '3': "0011",
                        '4': "0100",
                        '5': "0101",
                        '6': "0110",
                        '7': "0111",
                        '8': "1000",
                        '9': "1001",
                        '10': "1010",
                        '11': "1011",
                        '12': "1100",
                        '13': "1101",
                        '14': "1110",
                        '15': "1111"}
        convert_bin_hex_opp = {
            "0000": '0',
            "0001": '1',
            "0010": '2',
            "0011": '3',
            "0100": '4',
            "0101": '5',
            "0110": '6',
            "0111": '7',
            "1000": '8',
            "1001": '9',
            "1010": 'A',
            "1011": 'B',
            "1100": 'C',
            "1101": 'D',
            "1110": 'E',
            "1111": 'F'
        }




        initial_permutation_table = [
            58, 50, 42, 34, 26, 18, 10, 2,
            60, 52, 44, 36, 28, 20, 12, 4,
            62, 54, 46, 38, 30, 22, 14, 6,
            64, 56, 48, 40, 32, 24, 16, 8,
            57, 49, 41, 33, 25, 17, 9, 1,
            59, 51, 43, 35, 27, 19, 11, 3,
            61, 53, 45, 37, 29, 21, 13, 5,
            63, 55, 47, 39, 31, 23, 15, 7
        ]

        expansion_permutation_table = [
            32, 1, 2, 3, 4, 5,
            4, 5, 6, 7, 8, 9,
            8, 9, 10, 11, 12, 13,
            12, 13, 14, 15, 16, 17,
            16, 17, 18, 19, 20, 21,
            20, 21, 22, 23, 24, 25,
            24, 25, 26, 27, 28, 29,
            28, 29, 30, 31, 32, 1
        ]

        permutation_function_table = [
            16, 7, 20, 21, 29, 12, 28, 17,
            1, 15, 23, 26, 5, 18, 31, 10,
            2, 8, 24, 14, 32, 27, 3, 9,
            19, 13, 30, 6, 22, 11, 4, 25
        ]

        permuted_choice_one_table = [
            57, 49, 41, 33, 25, 17, 9,
            1, 58, 50, 42, 34, 26, 18,
            10, 2, 59, 51, 43, 35, 27,
            19, 11, 3, 60, 52, 44, 36,
            63, 55, 47, 39, 31, 23, 15,
            7, 62, 54, 46, 38, 30, 22,
            14, 6, 61, 53, 45, 37, 29,
            21, 13, 5, 28, 20, 12, 4
        ]

        key_rotation_table = {
            1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2,
            9: 1, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 1
        }

        permuted_choice_two_table = [
            14, 17, 11, 24, 1, 5,
            3, 28, 15, 6, 21, 10,
            23, 19, 12, 4, 26, 8,
            16, 7, 27, 20, 13, 2,
            41, 52, 31, 37, 47, 55,
            30, 40, 51, 45, 33, 48,
            44, 49, 39, 56, 34, 53,
            46, 42, 50, 36, 29, 32
        ]
        s_boxes = [
            # S1
            [
                [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
            ],
            # S2
            [
                [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
            ],
            # S3
            [
                [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
                [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
                [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
                [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
            ],
            # S4
            [
                [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
                [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
                [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
                [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
            ],
            # S5
            [
                [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
                [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
                [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
                [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
            ],
            # S6
            [
                [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
                [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
                [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
                [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
            ],
            # S7
            [
                [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
                [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
                [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
                [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
            ],
            # S8
            [
                [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
                [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
                [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
                [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
            ]
        ]

        inverse_initial_permutation_table = [
            40, 8, 48, 16, 56, 24, 64, 32,
            39, 7, 47, 15, 55, 23, 63, 31,
            38, 6, 46, 14, 54, 22, 62, 30,
            37, 5, 45, 13, 53, 21, 61, 29,
            36, 4, 44, 12, 52, 20, 60, 28,
            35, 3, 43, 11, 51, 19, 59, 27,
            34, 2, 42, 10, 50, 18, 58, 26,
            33, 1, 41, 9, 49, 17, 57, 25
        ]
        cipher_text=""
        input_binary = ''.join(convert_hex_bin[i] for i in inputt)
        key_binary = ''.join( convert_hex_bin[i] for i in key)
        key_binary_permutated=[]
        for index,bit in enumerate(input_binary):
            input_binary_permutated.append(int(input_binary[initial_permutation_table[index]-1]))

        for index in range(0,56):
            key_binary_permutated.append(int(key_binary[permuted_choice_one_table[index]-1]))



        left_halve=input_binary_permutated[:32] #32
        right_halve=input_binary_permutated[32:64] #32

        C0=key_binary_permutated[0:28]
        D0=key_binary_permutated[28:]

        for i in range(0,16):
            
            right_halve_temp=[]

            for index in range(0,48):
                right_halve_temp.append(int(right_halve[expansion_permutation_table[index]-1]))
            
            C0=rotate_left(C0,key_rotation_table[i+1])
            D0=rotate_left(D0,key_rotation_table[i+1])

            key=[]
            key.extend(C0)
            key.extend(D0)
            
            key_permuted_choice_two=[]

            for index in range(0,48):
                
                key_permuted_choice_two.append(int(key[permuted_choice_two_table[index]-1]))


            XOR_result= list(a^b for a,b in zip(key_permuted_choice_two,right_halve_temp))
            #sbox:
            XOR_result_temp=""
            sbox_nu = 0
            for bit_number in range(0,len(XOR_result),6):
                row = int(str(XOR_result[bit_number])+str(XOR_result[bit_number+5]), 2) #convert binary to decimal
                column = int(str(XOR_result[bit_number+1])+str(XOR_result[bit_number+2])+str(XOR_result[bit_number+3])+str(XOR_result[bit_number+4]), 2) #convert binary to decimal
                # print("row and col = ",row,column)
                XOR_result_temp += convert_hex_bin_1_15[str(s_boxes[sbox_nu][row][column])]
                sbox_nu += 1

            sbox_permutation=[]
            for index in range(0,len(XOR_result_temp)):
                sbox_permutation.append(int(XOR_result_temp[permutation_function_table[index]-1]))
            right_halve_t = list(a^b for a,b in zip(left_halve,sbox_permutation))
            left_halve = right_halve
            # cipher_text=left_halve+right_halve
            right_halve = right_halve_t
            cipher_text=right_halve+left_halve


        cipher_text_str=""
        cipher_text_final=""

        for index,bit in enumerate(cipher_text):
            cipher_text_str+=str(cipher_text[inverse_initial_permutation_table[index]-1])
        for i in range(0,len(cipher_text_str),4):
            cipher_text_final += convert_bin_hex_opp[cipher_text_str[i:i+4]]
        #"4E0E6864B5E1CA52"
    
        return cipher_text_final
    except Exception as e:
        print(f"An error occurred in sec_hard riddle solver with input = {input} and output = {cipher_text_final}, error = {e}")
        return ""


def count_words(words):
    result = {}
    for word in words:
        if word in result:
            result[word] += 1
        else:
            result[word] = 1
    return result
def solve_problem_solving_easy(input: tuple) -> list:
    """
    This function takes a tuple as input and returns a list as output.

    Parameters:
    input (tuple): A tuple containing two elements:
        - A list of strings representing a question.
        - An integer representing a key.

    Returns:
    list: A list of strings representing the solution to the problem.
    """
    sorted_words = sorted(count_words(input[0]).items(), key=lambda x: (x[0]), reverse=False) 
    sorted_words2= dict(sorted_words[:])
    # print(sorted_words2)
    sorted_words3 = sorted(sorted_words2.items(), key=lambda x: (x[1]), reverse=True)
    # print(sorted_words3)
    sol = list( dict(sorted_words3[:]).keys())
    # print(sol[:input[1]])
    return sol[:input[1]]


def solve_problem_solving_medium(input: str) -> str:
    """
    This function takes a string as input and returns a string as output.

    Parameters:
    input (str): A string representing the input data.

    Returns:
    str: A string representing the solution to the problem.
    """
    try:
        stack = []
        index = len(input) - 1
        stack.append(input[index])
        numbers = "0123456789"
        temp = ""
        flag = 0

        while(index != -1):
            temp = input[index]
            if(temp in numbers):
                ourNumber = ""
                while(temp in numbers):
                    index -= 1
                    ourNumber += temp
                    temp = input[index]
                ourNumber = ourNumber[::-1]
                stack.pop()
                temp = ""
                t = ""
                while (t != "]"):
                    t = stack.pop()
                    temp += t  # kda m3na el string el byn el brackets + ]
                    flag = 1
                if (flag == 1):
                    flag = 0
                    stack.append(temp[0:-1] * int(ourNumber))

            else:
                index -= 1
                stack.append(temp)

        output = ""
        for i in range(len(stack) - 1, 0, -1):
            output += stack[i]
        return output

    except Exception as e:
        print(f"An error occurred in problem solving riddle meduim of input = {input}  and output = {output} ")
    



def solve_problem_solving_hard(input: tuple) -> int:
    """
    This function takes a tuple as input and returns an integer as output.

    Parameters:
    input (tuple): A tuple containing two integers representing m and n.

    Returns:
    int: An integer representing the solution to the problem.
    """
    grid = [[0 for i in range(input[1])] for j in range(input[0])]
    for i in range(input[0]):
        for j in range(input[1]):
            if i == 0 or j == 0:
                grid[i][j] = 1
            else:
                grid[i][j] = grid[i-1][j] + grid[i][j-1]
    return grid[input[0]-1][input[1]-1]




riddle_solvers = {
    'cv_easy': solve_cv_easy,
    'cv_medium': solve_cv_medium,
    'cv_hard': solve_cv_hard,
    'ml_easy': solve_ml_easy,
    'ml_medium': solve_ml_medium,
    'sec_medium_stegano': solve_sec_medium,
    'sec_hard':solve_sec_hard,
    'problem_solving_easy': solve_problem_solving_easy,
    'problem_solving_medium': solve_problem_solving_medium,
    'problem_solving_hard': solve_problem_solving_hard
}
