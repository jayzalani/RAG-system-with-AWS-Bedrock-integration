import boto3
import json

prompt_data = """Act as the Pandit in Indian Society and give qualities for men and women that are required in indian marriages in Hinglish, Give a Detailed Answer """

bedrock = boto3.client(service_name = 'bedrock-runtime')

payload = {
    "prompt" : "[INST]" +  prompt_data+"[/INST]",
    "max_gen_len": 512,
    "temperature": 0.5,
    "top_p":0.9
}
body = json.dumps(payload)
model_id ="arn:aws:bedrock:us-east-1:014498626498:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"
response  = bedrock.invoke_model(
    body = body,
    modelId = model_id,
    contentType = "application/json",
    accept = "application/json"
)

response_body = json.loads(response.get("body").read())
response_text = response_body['generation']
print(response_text)