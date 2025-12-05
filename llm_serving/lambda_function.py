import json
import boto3
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# --- Configuration (Set as Lambda Environment Variables) ---
# NOTE: The actual values are pulled from the Lambda's Environment Variables.
# We use os.environ.get() here.
KNOWLEDGE_BASE_ID = os.environ.get('KNOWLEDGE_BASE_ID', 'D7LNYAEDOD')
REGION_NAME = os.environ.get('AWS_REGION',
                             'us-east-1')  # AWS automatically sets the region
MODEL_ARN = os.environ.get('MODEL_ARN',
                           'arn:aws:sagemaker:us-east-1:144675714287:endpoint/biomistral-7b-gguf')
# GUARDRAIL_ID = os.environ.get('GUARDRAIL_ID', 'gr-qtktcbjlc2rh')
# GUARDRAIL_VERSION = os.environ.get('GUARDRAIL_VERSION', 'DRAFT')

# Initialize the Bedrock Agent Runtime client
# boto3 will automatically use the Lambda's Execution Role permissions
bedrock_agent_runtime = boto3.client(
    service_name='bedrock-agent-runtime',
    region_name='us-east-1'
)


def retrieve_and_generate_response(query: str):
    """Calls the Bedrock Agent Runtime API for RAG (TEXT ONLY)."""

    request_params = {
        'input': {'text': query},
        'retrieveAndGenerateConfiguration': {
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': KNOWLEDGE_BASE_ID,
                'modelArn': MODEL_ARN
            }

            # This code isn't working
            # },
            # 'guardrailConfiguration': {
            #     'guardrailIdentifier': GUARDRAIL_ID,
            #     'guardrailVersion': GUARDRAIL_VERSION
        }
    }

    try:
        response = bedrock_agent_runtime.retrieve_and_generate(**request_params)

        generated_text = response['output']['text']

        citations = []
        if 'citations' in response and response['citations']:
            for citation in response['citations']:
                if 'retrievedReferences' in citation:
                    for ref in citation['retrievedReferences']:
                        # Extract the content and the source URI (S3 path)
                        source_uri = ref.get('uri', 'N/A: URI Missing')
                        citations.append({
                            'content': ref['content']['text'],
                            'source_uri': source_uri
                        })

        short_answer_match = re.split(r'[.!?]', generated_text, 1)
        short_answer = short_answer_match[0].strip() + '.' if \
        short_answer_match[0].strip() else "No short answer available."

        certainty_score = "High" if citations else "Low/No Context"

        return {
            'answer': generated_text,
            'short_answer': short_answer,
            'certainty': certainty_score,
            'citations': citations
        }

    except Exception as e:
        print(f"❌ ERROR calling Bedrock API: {e}")
        # Re-raise the exception or return a structured error for debugging
        raise Exception(f"Bedrock RAG execution failed: {str(e)}")


def lambda_handler(event, context):
    """
    Standard AWS Lambda entry point. Handles the API Gateway request.
    """

    # 1. Parse the incoming user query from the API Gateway event
    try:
        # API Gateway sends the input JSON as a string in the 'body' field
        body = json.loads(event.get('body', '{}'))
        user_prompt = body.get("prompt")

        if not user_prompt:
            return {
                'statusCode': 400,
                'body': json.dumps(
                    {'error': 'Missing "prompt" in request body.'})
            }

    except Exception as e:
        print(f"Input parsing error: {e}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Invalid JSON format: {str(e)}'})
        }

    # 2. Call the RAG function with the user's query
    try:
        rag_result = retrieve_and_generate_response(user_prompt)

        # 3. Return the successful response back to API Gateway
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps(rag_result)
        }

    except Exception as e:
        # Catch and handle errors from the RAG function
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }