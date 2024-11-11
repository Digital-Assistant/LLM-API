import json
import boto3

# Bedrock client used to interact with APIs around models
bedrock = boto3.client(
    service_name='bedrock', 
    region_name='us-west-2'
)

# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'
)

def lambda_handler(event, context):
    # Specify model we want to use
    #modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    modelId = "anthropic.claude-3-haiku-20240307-v1:0"
    # Parse the incoming POST data
    try:
        # Check if the body is in the event
        if 'body' in event:
            print ("Body found in event")
            # If the body is a string, parse it as JSON
            if isinstance(event['body'], str):
                post_data = json.loads(event['body'])
            else:
                post_data = event['body']
        else:
            if isinstance(event, str):
                post_data = json.loads(event)
            else:
                post_data = event

        # Extract the user's question from the POST data
        #user_question = post_data.get('question', 'What is an AWS Lambda function?')
        print("Extracting UST")
        ust = post_data["UST"]
        
        print (ust)
        labels_with_ids = []
        print ("Extracting recordings")
        ii = 0
        for recording in post_data["Recordings"]:
            for label in recording["Recording_Labels"]:
                labels_with_ids.append({
                    "serial_id": ii,
                    "recording_id": recording["Recording_Id"],
                    "label": label
                })
            ii+=1
        list_of_recordings = json.dumps(labels_with_ids, indent=2)
        print ("list of recordings")
        print (list_of_recordings)
        # Generate the LLM prompt
        prompt = f"""
        Given the following User Search Term (UST):
        "{ust}"
        
        And the following list of recording labels with their IDs:
        {list_of_recordings}
        
        Please select the best possible label to answer the UST. Consider the semantic meaning and relevance of each label to the UST.
        
        Provide your answer in the following format:
        {{
            "selected_index": <serial id of the selected recording>,
            "selected_recording_id": <recording id of the selected recording>,
            "selected_label": "<the selected label>",
        }}
        
        Only provide the JSON object as your response, without any additional text.
        """
        
        body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2049,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "top_p": 1
    })
    
        # The call made to the model
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=modelId,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        
        # The response from the model
        answer = response_body.get("content")[0].get("text")
        
        
        selected_recording_id = json.loads(answer)["selected_recording_id"]
        selected_index = json.loads(answer)["selected_index"]
        print("Best Recording is :-")
        print (selected_recording_id)
        print("Best Recording Index is :-")
        print (selected_index)     
        
        
        best_recording = post_data["Recordings"][int(selected_index)]
        inputs = [input_data["Input"] for input_data in best_recording["Expected_User_Input"]]
        print("Inputs for the first recording:", inputs)
    except json.JSONDecodeError as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Invalid JSON in request body: {str(e)}'})
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Error processing request: {str(e)} : {event}'})
        }

   
    
    prompt = f"""
    Given the following user query:
    "{ust}"
    
    Please extract the following variables:
    {', '.join(inputs)}
    
    For each variable, provide the extracted value or 'Not found' if the information is not present in the query.
    
    Format your response as a Python dictionary, like this:
    {{
        "Project_Name": <extracted value or "Not found">,
        "User_Name": <extracted value or "Not found">
    }}
    
    Only provide the dictionary as your response, without any additional explanation.
    """
    
    # The payload provided to Bedrock 
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2049,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "top_p": 1
    })
    
    # The call made to the model
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=modelId,
        accept='application/json',
        contentType='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    
    # The response from the model
    answer = response_body.get("content")[0].get("text")
    print("Here is the answer:")
    print(answer)
    updated_answer = json.loads(answer)
    updated_answer["selected_recording_id"] = selected_recording_id
    
    # Convert the dictionary back to a JSON string
    #updated_answer = json.dumps(updated_answer, indent=4)
    
    return {
        'statusCode': 200,
        'body': json.dumps({ "Answer": updated_answer })
    }


def main():
    dummy_event = {"UST":"How much time did abhi spend on Udan","Recordings":[{"Recording_Id":4531,"Recording_Labels":["How much time did a user spent on project","Time spent by a user on project","user spent time on a project"],"Expected_User_Input":[{"Input":"project_name","input_metadata":""},{"Input":"action","input_metadata":""},{"Input":"user_name","input_metadata":""}]}]}
    dummy_event1 = {
    "UST": "How much time did ajay spent on udan",
    "Recordings": [
        {
        "Recording_Id": 4531,
        "Recording_Labels": [
            "How much time did a user spent on project",
            "Time spent by a user on project",
            "user spent time on a project"
        ],
        "Expected_User_Input": [
            {
            "Input": "project_name",
            "input_metadata": ""
            },
            {
            "Input": "action",
            "input_metadata": ""
            },
            {
            "Input": "user_name",
            "input_metadata": ""
            }
        ]
        },
        {
        "Recording_Id": 4530,
        "Recording_Labels": [
            "Time spent by user in a project",
            "Time spent by Ajay in Digital Assistant",
            "Get time spent by ajay in digital assistant"
        ],
        "Expected_User_Input": [
            {
            "Input": "project_name",
            "input_metadata": ""
            },
            {
            "Input": "action",
            "input_metadata": ""
            },
            {
            "Input": "user_filter_selector",
            "input_metadata": ""
            },
            {
            "Input": "user_name",
            "input_metadata": ""
            }
        ]
        }
    ]
    }
    
    # Call lambda_handler with dummy event
    result = lambda_handler(dummy_event, None)
    print("Lambda handler result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()