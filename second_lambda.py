import json
import boto3
import sys
import ast

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

#modelId = "anthropic.claude-3-opus-20240229-v1:0"
#modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
modelId = "anthropic.claude-3-haiku-20240307-v1:0"

def getRankedList(ust,list_of_recordings):
    prompt = f"""
        Given the following User Search Term (UST):
        "{ust}"
        
        And the following list of recording labels with their Serial IDs and Redocrding IDs:
        {list_of_recordings}
        
        Please rank all the labels from best to worst in terms of answering the UST. Consider the semantic meaning and relevance of each label to the UST.
        Provide your answer in the following format:
        [
            {{
                "rank": 1,
                "selected_index": <serial id of the recording>,
                "selected_recording_id": <recording id of the recording>,
                "selected_label": "<the selected label>",
            }},
            {{
                "rank": 2,
                ...
            }},
            ...
        ]
        Select only one label per recording id. Ie if there are 3 labels for a recording id, select only the best one.
        Include all labels in the ranked list. Only provide the JSON array as your response, without any additional text.
        """
        
    body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": prompt
        }
    ],
    "temperature": 0.2,
    "top_p": 1
    })
    
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId=modelId,
        accept='application/json',
        contentType='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    answer = response_body.get("content")[0].get("text")
    ranked_list = json.loads(answer)
    return ranked_list

def getInputValues(inputs, ust):

    prompt = f"""
    Given the following user query:
    "{ust}"
    
    Please extract the following variables:
    {', '.join(inputs)}
    
    For each variable, provide the extracted value if found in the query. If the information is not present, set "found" to "False" and "InputValue" to an empty string.
    
    Format your response as a JSON array of objects, like this:
    [
        {{
            "Input": "input_name",
            "found": "True",
            "InputValue": "extracted_value"
        }},
        {{
            "Input": "another_input",
            "found": "False",
            "InputValue": ""
        }},
        ...
    ]
    
    Only provide the JSON array as your response, without any additional explanation.
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
    #answer = response_body.get("content")[0].get("text")
    answer = response_body.get("content")[0].get("text")
    answer = json.loads(answer)
    print("Here is the answer:")
    print (answer)
    
    return answer
    #return json.dumps(result, indent=2)
def getStitchedResponse(ust, matched_recordings):
    formatted_json = {
        "UST": ust,
        "matched_recordings": []
    }

    for recording in matched_recordings:
        input_values = recording["inputValues"]
        #input_values = ast.literal_eval(recording["inputValues"])
        formatted_recording = {
            "recording_id": recording["recording_id"],
            "matched_recording_label": recording["matched_recording_label"],
            "inputValues": input_values
            #"inputValues": recording["inputValues"]
            #"inputValues": "[{name:abhi, rool: 1}, {name:def, rool: 2}, {name:ghi, rool: 3}]"
        }
        formatted_json["matched_recordings"].append(formatted_recording)

    return json.dumps(formatted_json)
def simplifyJson(original_json_string):
    parsed_json = json.loads(original_json_string)

    # Parse the nested inputValues string
    for recording in parsed_json['matched_recordings']:
        recording['inputValues'] = json.loads(recording['inputValues'])

    # Create a simplified JSON string
    simplified_json = json.dumps(parsed_json, indent=2)

    return simplified_json

def lambda_handler(event, context):
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

        print("Extracting UST")
        ust = post_data["UST"]
        
        print (ust)
        labels_with_ids = []
        print ("Extracting recordings...")
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
        #print ("list of recordings")
        #print (list_of_recordings)

        print ("Getting ranked list of labels...")
        ranked_list = getRankedList(ust,list_of_recordings)
        
        #print("Ranked list of recordings:")
        #print(json.dumps(ranked_list, indent=2))
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
    matched_recordings = []
    seen_recording_ids = set()
    for rank, item in enumerate(ranked_list, 1):
            selected_index = item['selected_index']
            selected_recording_id = item['selected_recording_id']
            selected_label = item['selected_label']
            if selected_recording_id in seen_recording_ids:
                continue
            seen_recording_ids.add(selected_recording_id)

            a_recording = post_data["Recordings"][int(selected_index)]
            inputs = [input_data["Input"] for input_data in a_recording["Expected_User_Input"]]
            print("getting input values...")
            inputValues = getInputValues(inputs, ust)
            matched_recordings.append({
                "recording_id": selected_recording_id,
                "matched_recording_label": selected_label,
                "inputValues": inputValues
            })
            '''print(f"Rank {rank}:")
            print(f"  Selected Index: {selected_index}")
            print(f"  Recording ID: {selected_recording_id}")
            print(f"  Label: {selected_label}")
            print(f"  Inputs: {inputs}")
            print(f"  Input Values: {inputValues}")
            print()
            '''

    #print(f"Matched Recordings: {matched_recordings}")
    response = getStitchedResponse(ust, matched_recordings)
    print ("Stitched response")
    print (response)
    print ("After Stringification")
    print (json.dumps(response))
    #response = simplifyJson(response)

    #print(f"Stitched Response: {response}")    
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }


def main():
    dummy_event = {
        "UST": "Show me Spent time on Digital Assistant by Yuresh",
        "Recordings": [
            {
                "Recording_Id": 4531,
                "Recording_Labels": [
                    "See Spent time on Udan by Abhishek"
                ],
                "Expected_User_Input": [
                    {
                        "Input": "ProjectName",
                        "input_metadata": "This is the name of the Project"
                    },
                    {
                        "Input": "ProjectNaviation",
                        "input_metadata": "This is the Action to take"
                    },
                    {
                        "Input": "UserCondition",
                        "input_metadata": "This is used for selecting the user condition",
                        "html_element_type": "dropdown",
                        "html_content": "<select id=\"operators_assigned_to_id\" name=\"op[assigned_to_id]\"><option value=\"=\">is</option><option value=\"!\">is not</option><option value=\"!\">none</option><option value=\"\">any</option></select>"
                    },
                    {
                        "Input":"UserName",
                        "input_metadata": "This is the user name dropdown",
                        "html_element_type": "dropdown",
                        "html_content": "<select class=\"value\" id=\"values_assigned_to_id_1\" name=\"v[assigned_to_id][]\"><option value=\"me\"><< me >></option><optgroup label=\"active\"><option value=\"86\">Aakash Entab</option><option value=\"99\">Abhishek Mathur</option><option value=\"8\">ajay k</option><option value=\"84\">Bhushan Entab</option><option value=\"87\">Ganesh Chandu</option><option value=\"12\">Haritha C</option><option value=\"83\">Jitendar Kumar</option><option value=\"82\">Jitendar Sharma</option><option value=\"64\">Lakshman Veti</option><option value=\"45\">Moen Ediga</option><option value=\"22\">Nagamunemma T</option><option value=\"88\">Navya Nimmagadda</option><option value=\"81\">Raju Kamireddy</option><option value=\"5\">Ramakrishna Krishnamsetty</option><option value=\"85\">Sandhya Entab</option><option value=\"65\">TBD TBD</option><option value=\"75\">Tej Reddy</option><option value=\"36\">Yureshwar Ravuri</option></optgroup><optgroup label=\"locked\"><option value=\"76\">Amith Bachuwala</option><option value=\"71\">ashwini indukande</option><option value=\"66\">Atul Arora</option><option value=\"9\">Dharani Reddy</option><option value=\"68\">Praveen Dodda</option><option value=\"54\">Ragavan K</option><option value=\"70\">Rama Krishna Mundru</option><option value=\"69\">Sunil Gutta</option><option value=\"102\">Udan Public</option></optgroup><option value=\"36\">Yureshwar Ravuri</option></select>"
                    }
                ]
            }
        ]
    }
    
    #dummy_event = {"UST":"How much time did abhi spend on Udan","Recordings":[{"Recording_Id":4531,"Recording_Labels":["How much time did a user spent on project","Time spent by a user on project","user spent time on a project"],"Expected_User_Input":[{"Input":"project_name","input_metadata":""},{"Input":"action","input_metadata":""},{"Input":"user_name","input_metadata":""}]}]}
    
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
    print(result)

if __name__ == "__main__":
    main()
