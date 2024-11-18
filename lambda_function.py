import json
import boto3
import sys
import ast
import test_anthropic3
import os

USE_DIRECT_ANTHROPIC = True

# Bedrock Runtime client used to invoke and question the models
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2'
)

#modelId = "anthropic.claude-3-opus-20240229-v1:0"
modelIdNonTrivial = "anthropic.claude-3-5-sonnet-20240620-v1:0"
modelIdTrivial = "anthropic.claude-3-5-haiku-20241022-v1:0"
#modelIdTrivial = "anthropic.claude-3-5-sonnet-20240620-v1:0"
#modelId = "anthropic.claude-3-haiku-20240307-v1:0"

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
        
    if USE_DIRECT_ANTHROPIC:
        # Initialize Anthropic client
        client = test_anthropic3.SimpleAnthropicClient(os.getenv("ANTHROPIC_API_KEY"))
        response = client.create_message(content=prompt)
        answer = response["content"][0]["text"]
        return json.loads(answer)
    else:
        # Existing Bedrock implementation
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 1
        })
        
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=modelIdNonTrivial,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        answer = response_body.get("content")[0].get("text")
        return json.loads(answer)
def getInputValuesWithBB(inputs, ust):
    formatted_inputs = []
    for input_dict in inputs:
        for input_name, input_data in input_dict.items():
            html_content = json.loads(input_data["html_content"])
            input_metadata = input_data["input_metadata"]
            options = []
            if '<select' in html_content:
                # Extract all option values and texts using basic string operations
                option_parts = html_content.split('<option')
                for part in option_parts[1:]:  # Skip first split as it's before first option
                    if 'value="' in part:
                        id = part.split('value="')[1].split('"')[0]
                        # Get text between > and </option>
                        value_text = part.split('>')[1].split('</option')[0] if '</option' in part else value
                        options.append({"possible_value_id": id, "possible_value_text": value_text})
            
            formatted_inputs.append({
                "input_name": input_name,
                "possible_values": options,
                "input_metadata": input_metadata
            })

    prompt = f"""
    Given the following user query:
    "{ust}"
    
    And the following input_name's with their possible values:
    {json.dumps(formatted_inputs, indent=2)}
    
    Please search for the most appropriate possible_value_text for each input_name based on the user query. Use the input_metadata to help you understand the context of the input_name.
    And return the corresponding possible_value_id for each input_name.
    Be very strict when it comes to matching Human Names.
    If no appropriate possible_value_text is found from the possible_values, set "found" to "False" and "InputValue" to an empty string.
    
    Format your response as a JSON array of objects, like this:
    [
        {{
            "Input": "input_name",
            "found": "True",
            "InputValue": "extracted possible_value_id"
        }},
        ...
    ]
    
    Only provide the JSON array as your response, without any additional explanation.
    """

    if USE_DIRECT_ANTHROPIC:
        client = test_anthropic3.SimpleAnthropicClient(os.getenv("ANTHROPIC_API_KEY"))
        response = client.create_message(content=prompt)
        answer = response["content"][0]["text"]
        return json.loads(answer)
    else:
        # Existing Bedrock implementation
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2049,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "top_p": 1
        })
        
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=modelIdNonTrivial,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        answer = response_body.get("content")[0].get("text")
        return json.loads(answer)

def getInputValuesWithoutBB(inputs, ust):
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
        ...
    ]
    
    Only provide the JSON array as your response, without any additional explanation.
    """

    if USE_DIRECT_ANTHROPIC:
        client = test_anthropic3.SimpleAnthropicClient(os.getenv("ANTHROPIC_API_KEY"))
        response = client.create_message(content=prompt)
        answer = response["content"][0]["text"]
        return json.loads(answer)
    else:
        # Existing Bedrock implementation
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2049,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "top_p": 1
        })
        
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId=modelIdTrivial,
            accept='application/json',
            contentType='application/json'
        )
        
        response_body = json.loads(response.get('body').read())
        answer = response_body.get("content")[0].get("text")
        return json.loads(answer)
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

def rerankMatchedRecordingsBasedOnInputsFound(matched_recordings):
    # Sort recordings based on number of found inputs (True values)
    return sorted(matched_recordings, 
                 key=lambda x: sum(1 for input_value in x['inputValues'] 
                                 if input_value['found'] == 'True'),
                 reverse=True)

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
            inputs_without_bb = [input_data["Input"] for input_data in a_recording["Expected_User_Input"] if "html_content" not in input_data]
            
            inputWithoutBBValues = getInputValuesWithoutBB(inputs_without_bb, ust) if inputs_without_bb else []
            #inputs_with_bb = [{input_data["Input"]: input_data["html_content"]} for input_data in a_recording["Expected_User_Input"] if "html_content"  in input_data]
            inputs_with_bb = [{
                input_data["Input"]: {
                    "html_content": input_data["html_content"],
                    "input_metadata": input_data.get("input_metadata", "")
                }
            } for input_data in a_recording["Expected_User_Input"] if "html_content" in input_data]

            inputWithBBValues = getInputValuesWithBB(inputs_with_bb, ust)
            inputValues = inputWithoutBBValues + inputWithBBValues
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
    matched_recordings = rerankMatchedRecordingsBasedOnInputsFound(matched_recordings)
    response = getStitchedResponse(ust, matched_recordings)
    print ("Stitched response")
    print (response)
    print ("After Stringification")
    print (json.dumps(response))
    #response = simplifyJson(response)

    #print(f"Stitched Response: {response}")    
    return {
        'statusCode': 200,
        'body': response
    }


def main():
    dummy_event = {"UST":"show me issues assigned to Abhishek","Recordings":[{"Recording_Id":4736,"Recording_Labels":["Show me issues for Yuresh"],"Expected_User_Input":[{"Input":"dropdown","input_metadata":"This is for selecting the drop list for users","html_element_type":"dropDown","html_content":"\"<select id=\\\"add_filter_select\\\"><option value=\\\"\\\">&nbsp;</option>\\n<option value=\\\"status_id\\\" disabled=\\\"disabled\\\">Status</option>\\n<option value=\\\"tracker_id\\\">Tracker</option>\\n<option value=\\\"priority_id\\\">Priority</option>\\n<option value=\\\"author_id\\\">Author</option>\\n<option value=\\\"assigned_to_id\\\" disabled=\\\"disabled\\\">Assignee</option>\\n<option value=\\\"fixed_version_id\\\">Target version</option>\\n<option value=\\\"category_id\\\">Category</option>\\n<option value=\\\"subject\\\">Subject</option>\\n<option value=\\\"description\\\">Description</option>\\n<option value=\\\"done_ratio\\\">% Done</option>\\n<option value=\\\"is_private\\\">Private</option>\\n<option value=\\\"attachment\\\">File</option>\\n<option value=\\\"watcher_id\\\">Watcher</option>\\n<option value=\\\"updated_by\\\">Updated by</option>\\n<option value=\\\"last_updated_by\\\">Last updated by</option>\\n<option value=\\\"subproject_id\\\">Subproject</option>\\n<option value=\\\"issue_id\\\">Issue</option><optgroup label=\\\"Assignee\\\"><option value=\\\"member_of_group\\\">Assignee's group</option>\\n<option value=\\\"assigned_to_role\\\">Assignee's role</option></optgroup><optgroup label=\\\"Target version\\\"><option value=\\\"fixed_version.due_date\\\">Target version's Due date</option>\\n<option value=\\\"fixed_version.status\\\">Target version's Status</option></optgroup><optgroup label=\\\"Date\\\"><option value=\\\"created_on\\\">Created</option>\\n<option value=\\\"updated_on\\\">Updated</option>\\n<option value=\\\"closed_on\\\">Closed</option>\\n<option value=\\\"start_date\\\">Start date</option>\\n<option value=\\\"due_date\\\">Due date</option></optgroup><optgroup label=\\\"Time tracking\\\"><option value=\\\"estimated_hours\\\">Estimated time</option>\\n<option value=\\\"spent_time\\\">Spent time</option></optgroup><optgroup label=\\\"Project\\\"><option value=\\\"project.status\\\">Project's Status</option></optgroup><optgroup label=\\\"Relations\\\"><option value=\\\"relates\\\">Related to</option>\\n<option value=\\\"duplicates\\\">Is duplicate of</option>\\n<option value=\\\"duplicated\\\">Has duplicate</option>\\n<option value=\\\"blocks\\\">Blocks</option>\\n<option value=\\\"blocked\\\">Blocked by</option>\\n<option value=\\\"precedes\\\">Precedes</option>\\n<option value=\\\"follows\\\">Follows</option>\\n<option value=\\\"copied_to\\\">Copied to</option>\\n<option value=\\\"copied_from\\\">Copied from</option>\\n<option value=\\\"parent_id\\\">Parent task</option>\\n<option value=\\\"child_id\\\">Subtasks</option></optgroup></select>\""},{"Input":"condition","input_metadata":"This is for applying the condition for certain user condition","html_element_type":"dropDown","html_content":"\"<select id=\\\"operators_assigned_to_id\\\" name=\\\"op[assigned_to_id]\\\"><option value=\\\"=\\\">is</option><option value=\\\"!\\\">is not</option><option value=\\\"!\\\">none</option><option value=\\\"\\\">any</option></select>\""},{"Input":"dropdown_user","input_metadata":"This is for selecting user from the list","html_element_type":"dropDown","html_content":"\"<select class=\\\"value\\\" id=\\\"values_assigned_to_id_1\\\" name=\\\"v[assigned_to_id][]\\\"><option value=\\\"me\\\">&lt;&lt; me &gt;&gt;</option><optgroup label=\\\"active\\\"><option value=\\\"86\\\">Aakash Entab</option><option value=\\\"99\\\">Abhishek Mathur</option><option value=\\\"8\\\">ajay k</option><option value=\\\"84\\\">Bhushan Entab</option><option value=\\\"87\\\">Ganesh Chandu</option><option value=\\\"12\\\">Haritha C</option><option value=\\\"83\\\">Jitendar Kumar</option><option value=\\\"82\\\">Jitendar Sharma</option><option value=\\\"64\\\">Lakshman Veti</option><option value=\\\"45\\\">Moen Ediga</option><option value=\\\"22\\\">Nagamunemma T</option><option value=\\\"88\\\">Navya Nimmagadda</option><option value=\\\"81\\\">Raju Kamireddy</option><option value=\\\"5\\\">Ramakrishna Krishnamsetty</option><option value=\\\"85\\\">Sandhya Entab</option><option value=\\\"65\\\">TBD TBD</option><option value=\\\"75\\\">Tej Reddy</option><option value=\\\"36\\\">Yureshwar Ravuri</option></optgroup><optgroup label=\\\"locked\\\"><option value=\\\"76\\\">Amith Bachuwala</option><option value=\\\"71\\\">ashwini indukande</option><option value=\\\"66\\\">Atul Arora</option><option value=\\\"9\\\">Dharani Reddy</option><option value=\\\"68\\\">Praveen Dodda</option><option value=\\\"54\\\">Ragavan K</option><option value=\\\"70\\\">Rama Krishna Mundru</option><option value=\\\"69\\\">Sunil Gutta</option><option value=\\\"102\\\">Udan Public</option></optgroup><option value=\\\"36\\\">Yureshwar Ravuri</option></select>\""}]},{"Recording_Id":4551,"Recording_Labels":["Show me issues in Features project","Navigate me to Features project issues","Features project issues"],"Expected_User_Input":[{"Input":"ProjectTitle","input_metadata":"This is for project name","html_element_type":"link","html_content":"\"<a class=\\\"project child leaf\\\" href=\\\"/projects/features\\\">Features</a>\""},{"Input":"ProjectNavigation","input_metadata":"This is for navigating under project","html_element_type":"link","html_content":"\"<a class=\\\"issues\\\" href=\\\"/projects/features/issues\\\">Issues</a>\""}]},{"Recording_Id":4543,"Recording_Labels":["Show me assigned issues of ajay in Digital Assistant project"],"Expected_User_Input":[{"Input":"project_name","input_metadata":"This is for project name field","html_element_type":"link","html_content":"\"<a class=\\\"project root parent\\\" href=\\\"/projects/digital-assistant\\\">Digital Assistant</a>\""},{"Input":"task","input_metadata":"this is for what kind of category","html_element_type":"link","html_content":"\"<a class=\\\"issues\\\" href=\\\"/projects/digital-assistant/issues\\\">Issues</a>\""},{"Input":"user_name","input_metadata":"this is for selecting username","html_element_type":"dropDown","html_content":"\"<select class=\\\"value\\\" id=\\\"values_assigned_to_id_1\\\" name=\\\"v[assigned_to_id][]\\\"><option value=\\\"me\\\">&lt;&lt; me &gt;&gt;</option><optgroup label=\\\"active\\\"><option value=\\\"86\\\">Aakash Entab</option><option value=\\\"99\\\">Abhishek Mathur</option><option value=\\\"8\\\">ajay k</option><option value=\\\"84\\\">Bhushan Entab</option><option value=\\\"87\\\">Ganesh Chandu</option><option value=\\\"12\\\">Haritha C</option><option value=\\\"83\\\">Jitendar Kumar</option><option value=\\\"82\\\">Jitendar Sharma</option><option value=\\\"64\\\">Lakshman Veti</option><option value=\\\"45\\\">Moen Ediga</option><option value=\\\"22\\\">Nagamunemma T</option><option value=\\\"88\\\">Navya Nimmagadda</option><option value=\\\"81\\\">Raju Kamireddy</option><option value=\\\"5\\\">Ramakrishna Krishnamsetty</option><option value=\\\"85\\\">Sandhya Entab</option><option value=\\\"65\\\">TBD TBD</option><option value=\\\"75\\\">Tej Reddy</option><option value=\\\"36\\\">Yureshwar Ravuri</option></optgroup><optgroup label=\\\"locked\\\"><option value=\\\"76\\\">Amith Bachuwala</option><option value=\\\"71\\\">ashwini indukande</option><option value=\\\"66\\\">Atul Arora</option><option value=\\\"9\\\">Dharani Reddy</option><option value=\\\"68\\\">Praveen Dodda</option><option value=\\\"54\\\">Ragavan K</option><option value=\\\"70\\\">Rama Krishna Mundru</option><option value=\\\"69\\\">Sunil Gutta</option><option value=\\\"102\\\">Udan Public</option></optgroup><option value=\\\"36\\\">Yureshwar Ravuri</option></select>\""}]},{"Recording_Id":4552,"Recording_Labels":["Navigate me to Activity in Digital Assistant","Show me activity in Digital Assistant"],"Expected_User_Input":[{"Input":"ProjectTitle","input_metadata":"This is a name of a Project","html_element_type":"link","html_content":"\"<a class=\\\"project root parent public\\\" href=\\\"/projects/digital-assistant\\\">Digital Assistant</a>\""},{"Input":"ProjectNavigation","input_metadata":"This is for navigating under project","html_element_type":"link","html_content":"\"<a class=\\\"activity\\\" href=\\\"/projects/digital-assistant/activity\\\">Activity</a>\""}]}]}
    #dummy_event = {"UST":"show me issues assigned to Mukesh","Recordings":[{"Recording_Id":4736,"Recording_Labels":["Show me issues for Yuresh"],"Expected_User_Input":[{"Input":"dropdown","input_metadata":"This is for selecting the drop list for users","html_element_type":"dropDown","html_content":"\"<select id=\\\"add_filter_select\\\"><option value=\\\"\\\">&nbsp;</option>\\n<option value=\\\"status_id\\\" disabled=\\\"disabled\\\">Status</option>\\n<option value=\\\"tracker_id\\\">Tracker</option>\\n<option value=\\\"priority_id\\\">Priority</option>\\n<option value=\\\"author_id\\\">Author</option>\\n<option value=\\\"assigned_to_id\\\" disabled=\\\"disabled\\\">Assignee</option>\\n<option value=\\\"fixed_version_id\\\">Target version</option>\\n<option value=\\\"category_id\\\">Category</option>\\n<option value=\\\"subject\\\">Subject</option>\\n<option value=\\\"description\\\">Description</option>\\n<option value=\\\"done_ratio\\\">% Done</option>\\n<option value=\\\"is_private\\\">Private</option>\\n<option value=\\\"attachment\\\">File</option>\\n<option value=\\\"watcher_id\\\">Watcher</option>\\n<option value=\\\"updated_by\\\">Updated by</option>\\n<option value=\\\"last_updated_by\\\">Last updated by</option>\\n<option value=\\\"subproject_id\\\">Subproject</option>\\n<option value=\\\"issue_id\\\">Issue</option><optgroup label=\\\"Assignee\\\"><option value=\\\"member_of_group\\\">Assignee's group</option>\\n<option value=\\\"assigned_to_role\\\">Assignee's role</option></optgroup><optgroup label=\\\"Target version\\\"><option value=\\\"fixed_version.due_date\\\">Target version's Due date</option>\\n<option value=\\\"fixed_version.status\\\">Target version's Status</option></optgroup><optgroup label=\\\"Date\\\"><option value=\\\"created_on\\\">Created</option>\\n<option value=\\\"updated_on\\\">Updated</option>\\n<option value=\\\"closed_on\\\">Closed</option>\\n<option value=\\\"start_date\\\">Start date</option>\\n<option value=\\\"due_date\\\">Due date</option></optgroup><optgroup label=\\\"Time tracking\\\"><option value=\\\"estimated_hours\\\">Estimated time</option>\\n<option value=\\\"spent_time\\\">Spent time</option></optgroup><optgroup label=\\\"Project\\\"><option value=\\\"project.status\\\">Project's Status</option></optgroup><optgroup label=\\\"Relations\\\"><option value=\\\"relates\\\">Related to</option>\\n<option value=\\\"duplicates\\\">Is duplicate of</option>\\n<option value=\\\"duplicated\\\">Has duplicate</option>\\n<option value=\\\"blocks\\\">Blocks</option>\\n<option value=\\\"blocked\\\">Blocked by</option>\\n<option value=\\\"precedes\\\">Precedes</option>\\n<option value=\\\"follows\\\">Follows</option>\\n<option value=\\\"copied_to\\\">Copied to</option>\\n<option value=\\\"copied_from\\\">Copied from</option>\\n<option value=\\\"parent_id\\\">Parent task</option>\\n<option value=\\\"child_id\\\">Subtasks</option></optgroup></select>\""},{"Input":"condition","input_metadata":"This is for applying the condition for certain user condition","html_element_type":"dropDown","html_content":"\"<select id=\\\"operators_assigned_to_id\\\" name=\\\"op[assigned_to_id]\\\"><option value=\\\"=\\\">is</option><option value=\\\"!\\\">is not</option><option value=\\\"!\\\">none</option><option value=\\\"\\\">any</option></select>\""},{"Input":"dropdown_user","input_metadata":"This is for selecting user from the list","html_element_type":"dropDown","html_content":"\"<select class=\\\"value\\\" id=\\\"values_assigned_to_id_1\\\" name=\\\"v[assigned_to_id][]\\\"><option value=\\\"me\\\">&lt;&lt; me &gt;&gt;</option><optgroup label=\\\"active\\\"><option value=\\\"86\\\">Aakash Entab</option><option value=\\\"99\\\">Abhishek Mathur</option><option value=\\\"8\\\">ajay k</option><option value=\\\"84\\\">Bhushan Entab</option><option value=\\\"87\\\">Ganesh Chandu</option><option value=\\\"12\\\">Haritha C</option><option value=\\\"83\\\">Jitendar Kumar</option><option value=\\\"82\\\">Jitendar Sharma</option><option value=\\\"64\\\">Lakshman Veti</option><option value=\\\"45\\\">Moen Ediga</option><option value=\\\"22\\\">Nagamunemma T</option><option value=\\\"88\\\">Navya Nimmagadda</option><option value=\\\"81\\\">Raju Kamireddy</option><option value=\\\"5\\\">Ramakrishna Krishnamsetty</option><option value=\\\"85\\\">Sandhya Entab</option><option value=\\\"65\\\">TBD TBD</option><option value=\\\"75\\\">Tej Reddy</option><option value=\\\"36\\\">Yureshwar Ravuri</option></optgroup><optgroup label=\\\"locked\\\"><option value=\\\"76\\\">Amith Bachuwala</option><option value=\\\"71\\\">ashwini indukande</option><option value=\\\"66\\\">Atul Arora</option><option value=\\\"9\\\">Dharani Reddy</option><option value=\\\"68\\\">Praveen Dodda</option><option value=\\\"54\\\">Ragavan K</option><option value=\\\"70\\\">Rama Krishna Mundru</option><option value=\\\"69\\\">Sunil Gutta</option><option value=\\\"102\\\">Udan Public</option></optgroup><option value=\\\"36\\\">Yureshwar Ravuri</option></select>\""}]}]}
    
    
    #dummy_event = {"UST":"show me issues for dharani","Recordings":[{"Recording_Id":4736,"Recording_Labels":["Show me issues for Yuresh"],"Expected_User_Input":[{"Input":"dropdown","input_metadata":"This is for selecting the drop list for users","html_element_type":"dropDown","html_content":"<select id=\"add_filter_select\"><option value=\"\">&nbsp;</option>\n<option value=\"status_id\" disabled=\"disabled\">Status</option>\n<option value=\"tracker_id\">Tracker</option>\n<option value=\"priority_id\">Priority</option>\n<option value=\"author_id\">Author</option>\n<option value=\"assigned_to_id\" disabled=\"disabled\">Assignee</option>\n<option value=\"fixed_version_id\">Target version</option>\n<option value=\"category_id\">Category</option>\n<option value=\"subject\">Subject</option>\n<option value=\"description\">Description</option>\n<option value=\"done_ratio\">% Done</option>\n<option value=\"is_private\">Private</option>\n<option value=\"attachment\">File</option>\n<option value=\"watcher_id\">Watcher</option>\n<option value=\"updated_by\">Updated by</option>\n<option value=\"last_updated_by\">Last updated by</option>\n<option value=\"subproject_id\">Subproject</option>\n<option value=\"issue_id\">Issue</option><optgroup label=\"Assignee\"><option value=\"member_of_group\">Assignee's group</option>\n<option value=\"assigned_to_role\">Assignee's role</option></optgroup><optgroup label=\"Target version\"><option value=\"fixed_version.due_date\">Target version's Due date</option>\n<option value=\"fixed_version.status\">Target version's Status</option></optgroup><optgroup label=\"Date\"><option value=\"created_on\">Created</option>\n<option value=\"updated_on\">Updated</option>\n<option value=\"closed_on\">Closed</option>\n<option value=\"start_date\">Start date</option>\n<option value=\"due_date\">Due date</option></optgroup><optgroup label=\"Time tracking\"><option value=\"estimated_hours\">Estimated time</option>\n<option value=\"spent_time\">Spent time</option></optgroup><optgroup label=\"Project\"><option value=\"project.status\">Project's Status</option></optgroup><optgroup label=\"Relations\"><option value=\"relates\">Related to</option>\n<option value=\"duplicates\">Is duplicate of</option>\n<option value=\"duplicated\">Has duplicate</option>\n<option value=\"blocks\">Blocks</option>\n<option value=\"blocked\">Blocked by</option>\n<option value=\"precedes\">Precedes</option>\n<option value=\"follows\">Follows</option>\n<option value=\"copied_to\">Copied to</option>\n<option value=\"copied_from\">Copied from</option>\n<option value=\"parent_id\">Parent task</option>\n<option value=\"child_id\">Subtasks</option></optgroup></select>"},{"Input":"condition","input_metadata":"This is for applying the condition for certain user condition","html_element_type":"dropDown","html_content":"<select id=\"operators_assigned_to_id\" name=\"op[assigned_to_id]\"><option value=\"=\">is</option><option value=\"!\">is not</option><option value=\"!\">none</option><option value=\"\">any</option></select>"},{"Input":"dropdown_user","input_metadata":"This is for selecting user from the list","html_element_type":"dropDown","html_content":"<select class=\"value\" id=\"values_assigned_to_id_1\" name=\"v[assigned_to_id][]\"><option value=\"me\">&lt;&lt; me &gt;&gt;</option><optgroup label=\"active\"><option value=\"86\">Aakash Entab</option><option value=\"99\">Abhishek Mathur</option><option value=\"8\">ajay k</option><option value=\"84\">Bhushan Entab</option><option value=\"87\">Ganesh Chandu</option><option value=\"12\">Haritha C</option><option value=\"83\">Jitendar Kumar</option><option value=\"82\">Jitendar Sharma</option><option value=\"64\">Lakshman Veti</option><option value=\"45\">Moen Ediga</option><option value=\"22\">Nagamunemma T</option><option value=\"88\">Navya Nimmagadda</option><option value=\"81\">Raju Kamireddy</option><option value=\"5\">Ramakrishna Krishnamsetty</option><option value=\"85\">Sandhya Entab</option><option value=\"65\">TBD TBD</option><option value=\"75\">Tej Reddy</option><option value=\"36\">Yureshwar Ravuri</option></optgroup><optgroup label=\"locked\"><option value=\"76\">Amith Bachuwala</option><option value=\"71\">ashwini indukande</option><option value=\"66\">Atul Arora</option><option value=\"9\">Dharani Reddy</option><option value=\"68\">Praveen Dodda</option><option value=\"54\">Ragavan K</option><option value=\"70\">Rama Krishna Mundru</option><option value=\"69\">Sunil Gutta</option><option value=\"102\">Udan Public</option></optgroup><option value=\"36\">Yureshwar Ravuri</option></select>"}]}]}
    
    
    dummy_event2 = {
        "UST": "Show me Spent time on Digital Assistant by BHalla",
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
