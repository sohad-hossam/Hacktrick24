import numpy as np
from LSBSteg import decode
import requests
import numpy as np
from LSBSteg import decode 
import torch
import pickle

api_base_url = 'http://3.70.97.142:5000'
team_id = 'UcXiSfT'


def init_eagle(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    payload = {'teamId': team_id}
    response = requests.post(api_base_url+'/eagle/start', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        footprints = response.json()['footprint']
        '''
        Footprints are returned as a map of 3 keys: ‘1’ ,’2’, ‘3’, each representing a channel
        number, in strings. The values will be an array representation of the footprints, which
        you should convert to a NumPy array to use.
        '''
        for key in footprints.keys():
            footprints[key] = np.array(footprints[key])
        return footprints
    else:
        return None




def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''
    
    model = torch.jit.load("C:/Users/sohad/Desktop/HackTrick24-main/Solvers/model1.pth")
    model.eval()
    max = 0
    max_index = -1
    counter = 0
    for key in footprint.keys(): 
        counter += 1
        with torch.no_grad():
            feature = np.where(np.isfinite(footprint[key]), footprint[key], 0)
            input = torch.tensor(feature).float()
            output = model(input.view(1, input.size(0), input.size(1)))
            _,predicted = torch.max(output, dim=2)
            predicted=predicted.view(output.shape[0]*output.shape[1])
            count_ones=torch.count_nonzero(predicted)
            if (max < count_ones) :
                
                max = count_ones
                max_index = key
    
    if (max < 10):
        return -1
    else:
        return int(max_index)        


  
def skip_msg(team_id):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    payload = {'teamId': team_id}
    response = requests.post(api_base_url + '/eagle/skip-message', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        next_footprints = response.json().get('nextFootprint')
        if next_footprints:
            for key in next_footprints.keys():
                next_footprints[key] = np.array(next_footprints[key])
            return next_footprints
        else:
            return None
    else:
        return None

  
def request_msg(team_id, channel_id):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    payload = {'teamId': team_id, 'channelId': channel_id}
    response = requests.post(api_base_url + '/eagle/request-message', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        encoded_msg = np.array(response.json()['encodedMsg'])
        return encoded_msg
    else:
        return None

def submit_msg(team_id, decoded_msg):
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    payload = {'teamId': team_id, 'decodedMsg': decoded_msg}
    response = requests.post(api_base_url + '/eagle/submit-message', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        next_footprints = response.json().get('nextFootprint')
        if next_footprints:
            return next_footprints
        else:
            return None
    else:
        return None
  
def end_eagle(team_id):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    payload = {'teamId': team_id}
    response = requests.post(api_base_url + '/eagle/end-game', json=payload)
    if response.status_code == 200 or response.status_code == 201:
        print(response.text)
        return response.text
    else:
        return None

def submit_eagle_attempt(team_id):
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as eagle
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''
    footprints = init_eagle(team_id)
    if footprints is None:
        return "Failed to initialize the game as an eagle"
    
    while(True):

        channel_id = select_channel(footprints)
        
        if channel_id == -1 :
            footprints = skip_msg(team_id)

        else:
            encoded_msg = request_msg(team_id, channel_id)
            if encoded_msg is None:
                return "Failed to request the message"
                
            decoded_msg = decode(encoded_msg)
            if decoded_msg is None:
                return "Failed to decode the message"
            
            footprints = submit_msg(team_id, decoded_msg)
        
        if footprints is None:
            break

    end_game = end_eagle(team_id)
    
    if end_game is None:
        return "Failed to end the game"
    else:
        return "Eagle game completed successfully"


submit_eagle_attempt(team_id)
