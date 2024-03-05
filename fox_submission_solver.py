import requests
import numpy as np
from LSBSteg import encode
from riddle_solvers import riddle_solvers
import random
import pickle
import json

api_base_url = 'http://3.70.97.142:5000'
team_id = 'UcXiSfT'

count = 0


def init_fox(team_id):
    print("inside fox")
    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''
    payload = {'teamId': team_id}
    response = requests.post(api_base_url+'/fox/start', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        message = data['msg']
        carrier_image = np.array(data['carrier_image'])
        # with open('message.pickle', 'wb') as file:
        #     pickle.dump(message, file)

        # with open('carrier_image.pickle', 'wb') as file:
        #     pickle.dump(carrier_image, file)
        
        return message, carrier_image
    else:
        return None, None


# def split_string_randomly(input_string, min_length=1, max_length=None):
#     if max_length is None:
#         max_length = len(input_string)

#     result = []
#     start_index = 0

#     while start_index < len(input_string):
#         end_index = start_index + random.randint(min_length, max_length)
#         if end_index > len(input_string):
#             end_index = len(input_string)
#         result.append(input_string[start_index:end_index])
#         start_index = end_index

#     return result


def generate_message_array(message, image_carrier, fake_message_budget):
    '''
    In this function you will need to create your own strategy. That includes:
        1. How you are going to split the real message into chunks
        2. Include any fake chunks
        3. Decide what 3 chunks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunk in the image carrier  
    '''

    # # split message to randomly sized bits
    # real_messages_array = split_string_randomly(message)
    # encode the randomly sized bits
    # real_messages_encoded = []
    # for message in real_messages_array:
    #/////////////////////////////////////////////////#
    chunck_size=(2*len(message))/fake_message_budget
    print("this is the chunck size: ",chunck_size)
    real_message_array = [message[i:i+int(chunck_size)] for i in range(0, len(message), int(chunck_size))]
    real_message_encoded=[]
    for i in range(len(real_message_array)):
        real_message_encoded.append(encode(image_carrier.copy(), real_message_array[i]))
        
    #/////////////////////////////////////////////////#
    
    ############ message will not be chuncked for testing###########
    # real_message = encode(image_carrier.copy(), message)



    ####################################################################
    # making fake messages

    fake_messages_encoded = []
    for i in range(fake_message_budget):
        # Select a random item from the array
        random_item1 = random.choice(real_message_array)
        random_item2 =random.choice(real_message_array)
        random_item3 =random.choice(real_message_array)

        fake_message = random_item1+random_item2+random_item3
        
        if(fake_message in real_message_array):
            fake_message = "NOTREAL"
        
        fake_message_encoded = encode(image_carrier.copy(), fake_message)
        fake_messages_encoded.append(fake_message_encoded)
    channel_data_index = [0, 1, 2]



    #//////////////////////////////////////////////////////////////////
    for real_message in real_message_encoded:
        random.shuffle(channel_data_index)
        messages = ['','', '']
        if len(fake_messages_encoded)>=2:
                messages[channel_data_index[0]] = real_message.tolist()
                messages[channel_data_index[1]] = fake_messages_encoded.pop().tolist()
                messages[channel_data_index[2]] = fake_messages_encoded.pop().tolist()
  


                message_entities = ['','','']
                message_entities[channel_data_index[0]] = 'R'
                message_entities[channel_data_index[1]] = 'F'
                message_entities[channel_data_index[2]] = 'F'


                send_message(team_id, messages, message_entities)   
                    


    #//////////////////////////////////////////////////////////////////
    # messages[channel_data_index[0]] = real_message_encoded.tolist()
    # messages[channel_data_index[1]] = fake_messages_encoded.pop().tolist()
    # messages[channel_data_index[2]] = fake_messages_encoded.pop().tolist()
    # print("this is the message array: ", messages , type(messages) )
    # print("messages ,THe real one: ", messages[0],type(messages[0]))
    # print("messages ,THe fake one: ", messages[1], type(messages[1]))
    # print("messages ,THe fake one: ", messages[2], type(messages[2]))   



    # message_entities = ['','','']
    # message_entities[channel_data_index[0]] = 'R'
    # message_entities[channel_data_index[1]] = 'F'
    # message_entities[channel_data_index[2]] = 'F'
    # print("message_entities: ", message_entities, type(message_entities))
    # print("message_entities first: ", message_entities[0], type(message_entities[0]))
    # print("message_entities second: ", message_entities[1], type(message_entities[1]))
    # print("message_entities third: ", message_entities[2], type(message_entities[2]))

    # send_message(team_id, messages, message_entities)


    # if fake_message_budget >= 2:
    #     messages = ["-", "-", "-"]
    #     messages[channel_data_index[0]] = real_message.tolist()
    #     messages[channel_data_index[1]] = fake_messages_encoded.pop().tolist()
    #     messages[channel_data_index[2]] = fake_messages_encoded.pop().tolist()
    #     message_entities = ["-", "-", "-"]
    #     message_entities[channel_data_index[0]] = 'R'
    #     message_entities[channel_data_index[1]] = 'F'
    #     message_entities[channel_data_index[2]] = 'F'
    #     send_message(team_id, messages, message_entities)
    # elif fake_message_budget == 1:
    #     messages = ["-", "-", "-"]
    #     messages[channel_data_index[0]] = real_message.tolist()
    #     messages[channel_data_index[1]] = fake_messages_encoded.pop().tolist()
    #     messages[channel_data_index[2]] = ''
    #     message_entities = ["-", "-", "-"]
    #     message_entities[channel_data_index[0]] = 'R'
    #     message_entities[channel_data_index[1]] = 'F'
    #     message_entities[channel_data_index[2]] = 'E'
    #     send_message(team_id, messages, message_entities)
    # else:
    #     messages = ["-", "-", "-"]
    #     messages[channel_data_index[0]] = real_message.tolist()
    #     messages[channel_data_index[1]] = ''
    #     messages[channel_data_index[2]] = ''
    #     message_entities = ["-", "-", "-"]
    #     message_entities[channel_data_index[0]] = 'R'
    #     message_entities[channel_data_index[1]] = 'E'
    #     message_entities[channel_data_index[2]] = 'E'
    #     send_message(team_id, messages, message_entities)



def get_riddle(team_id, riddle_id):
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that: 
        1. Once you requested a riddle you cannot request it again per game. 
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle. 
    '''
    payload = {'teamId': team_id, 'riddleId': riddle_id}
    response = requests.post(api_base_url+'/fox/get-riddle', json=payload)
    
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        riddle = data['test_case']
        
        try:
            riddle_string = json.dumps(data)
            with open('riddle'+str(count)+'.pickle', 'wb') as file:
                count += 1 
                pickle.dump(riddle_string, file)
        except Exception as e:
            print(f"error in riddle: {e}")
        finally:
            return riddle
    else:
        return None


def solve_riddle(team_id, solution):
    '''
    In this function you will solve the riddle that you have requested. 
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    '''
    payload = {'teamId': team_id, 'solution': solution}
    response = requests.post(api_base_url+'/fox/solve-riddle', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        budget_increase = data['budget_increase']
        total_budget = data['total_budget']
        status = data['status']

        print("budget increase:",budget_increase)
        print("total budget:",total_budget)
        print("budget function response",status)

        return budget_increase, total_budget, status
    else:
        return None, None, None


def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    '''
    Use this function to call the api end point to send one chunk of the message. 
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call. 
    '''
    payload = {'teamId': team_id, 'messages': messages,
               'message_entities': message_entities}
    response = requests.post(api_base_url+'/fox/send-message', json=payload)
    print("inside send message: ", response.status_code)
    if response.status_code == 200 or response.status_code == 201:
        data = response.json()
        status = data['status']
        print("this is the send message status",status)
        return status
    else:
        return None


def end_fox(team_id):
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire message within the timelimit of the game).
    '''
    payload = {'teamId': team_id}
    response = requests.post(api_base_url+'/fox/end-game', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        print(response.text)
        return response.text
    else:
        return None


def submit_fox_attempt(team_id):
    '''
     Call this function to start playing as a fox. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve riddles 
        3. Make your own Strategy of sending the messages in the 3 channels
        4. Make your own Strategy of splitting the message into chunks
        5. Send the messages 
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be atleast R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling 
    '''

    message, image_carrier = init_fox(team_id)
    if message is None or image_carrier is None:
        print("Failed to initialize the game as fox")
        return

    # fake messages count
    fake_messages_count = 0
    # riddle IDs
    PS_easy = "problem_solving_easy"
    PS_medium = "problem_solving_medium"
    PS_hard = "problem_solving_hard"
    CV_easy = "cv_easy"
    CV_medium = "cv_medium"
    ML_medium="ml_medium"
    S_hard="sec_hard"
    

####################################################
    # print("getting the ps easy riddle")
    # PS_easy_riddle_test_case = get_riddle(team_id, PS_easy)
    # PS_easy_solution = riddle_solvers["problem_solving_easy"](
    #     PS_easy_riddle_test_case)
    # budget_increase, total_budget, status = solve_riddle(
    #     team_id, PS_easy_solution)
    # print("riddle easy response: ",status)  

    # # update budget
    # if total_budget != 0:
    #     print("PS easy success")
    #     fake_messages_count = total_budget
    # else:
    #     print("PS easy fail")

####################################################
    print("getting the ps medium riddle")
    PS_medium_riddle_test_case = get_riddle(team_id, PS_medium)
    PS_medium_solution = riddle_solvers["problem_solving_medium"](
        PS_medium_riddle_test_case)
    budget_increase, total_budget, status = solve_riddle(
        team_id, PS_medium_solution)
    print("riddle medium response: ",status)    

    # update budget
    if total_budget is not None:
        print("PS medium success")
        fake_messages_count = total_budget
    else:
        print("PS medium fail")
#####################################################
    print("getting the ps hard riddle")    
    PS_hard_riddle_test_case = get_riddle(team_id, PS_hard)
    PS_hard_solution = riddle_solvers["problem_solving_hard"](
        PS_hard_riddle_test_case)
    budget_increase, total_budget, status = solve_riddle(
        team_id, PS_hard_solution)
    print("riddle hard response: ",status)

    # update budget
    if total_budget is not None:
        print("PS hard success")
        fake_messages_count = total_budget
    else:
        print("hard fail")
#####################################################
    # print("getting the cv easy riddle")
    # CV_easy_test_case = get_riddle(team_id, CV_easy)
    # CV_easy_solution = riddle_solvers["cv_easy"](
    #     CV_easy_test_case)
    # budget_increase, total_budget, status = solve_riddle(
    #     team_id, CV_easy_solution)
    # print("riddle easy response: ",status)

    # # update budget
    # if total_budget is not None:
    #     print("cv easy success")
    #     fake_messages_count = total_budget
    # else:
    #     print("cv easy fail")
#####################################################
    # CV_medium_test_case = get_riddle(team_id, CV_medium)
    # CV_medium_solution = riddle_solvers["cv_medium"](
    #     CV_medium_test_case)
    # budget_increase, total_budget, status = solve_riddle(
    #     team_id, CV_medium_solution)

    # # update budget
    # if total_budget is not None:
    #     print("cv medium success")
    #     fake_messages_count = total_budget
    # else:
    #     print("cv medium fail")
#####################################################
    print("getting the ml medium riddle")    
    ML_medium_test_case = get_riddle(team_id, ML_medium)
    ML_medium_solution = riddle_solvers["ml_medium"](
        ML_medium_test_case)
    budget_increase, total_budget, status = solve_riddle(
        team_id, ML_medium_solution)
    print("riddle medium ml response: ",status)

    # update budget
    if total_budget is not None:
        print("ml medium success")
        fake_messages_count = total_budget
    else:
        print("ml medium fail")        
        
#####################################################
         ######################################################
    print("getting the ml medium riddle")    
    S_hard_test_case = get_riddle(team_id, S_hard)
    S_hard_solution = riddle_solvers["sec_hard"](
        S_hard_test_case)
    budget_increase, total_budget, status = solve_riddle(
        team_id, S_hard_solution)
    print("riddle medium ml response: ",status)

    # update budget
    if total_budget is not None:
        print("ml medium success")
        fake_messages_count = total_budget
    else:
        print("ml medium fail")               
        
#####################################################
#####################################################        
    print("this is the fake messages count",fake_messages_count)

    generate_message_array(message, image_carrier, fake_messages_count)

    end_status = end_fox(team_id)
    if end_status is None:
        print("Failed to end the game")
        return
    print("this is the end of the game",end_status)


submit_fox_attempt(team_id)
