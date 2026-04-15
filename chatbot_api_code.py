import boto3
import json

client_sme = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    user_input = event.get('prompt', '')
    print(user_input)

    system_prompt = [
        {
            "text": "Act as a wind turbine manufacturing assistant. Summarize the logs in 5 lines."
        }
    ]

    message_list = [
        {
            "role": "user",
            "content": [{"text": user_input}]
        }
    ]

    inference_params = {
        "maxTokens": 2500,
        "topP": 0.9,
        "topK": 20,
        "temperature": 0.7
    }

    request_body = {
        "schemaVersion": "messages-v1",
        "messages": message_list,
        "system": system_prompt,
        "inferenceConfig": inference_params,
    }

    response = client_sme.invoke_model(
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json',
        # modelId='anthropic.claude-sonnet-4-6',
        modelId='arn:aws:bedrock:eu-central-1:864456252731:inference-profile/eu.amazon.nova-2-lite-v1:0'
        # modelId='amazon.nova-pro-v1:0',
        # trace='ENABLED',
        # guardrailIdentifier='string',
        # guardrailVersion='string',
        # performanceConfigLatency='standard',
    )

    print(response)

    # Read response body
    response_body = json.loads(response['body'].read().decode('utf-8'))
    generated_text = response_body['output']['message']['content'][0]['text']

    print(generated_text)


    return {
        'statusCode': 200,
        'body': json.dumps({
            "input": user_input,
            "response": generated_text
        })
    }
