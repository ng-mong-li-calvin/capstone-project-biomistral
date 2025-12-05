import requests
import json
import time

# 1. REPLACE THIS WITH YOUR ACTUAL API GATEWAY INVOKE URL
API_URL = "https://oqn6cgfx29.execute-api.ap-southeast-2.amazonaws.com/prod/query"

# 2. Define the test question payload
payload = {
    "prompt": "What daily dose of Vitamin C is recommended to reduce the \
    prevalence of complex regional pain syndrome after wrist fractures?"
}

# 3. Send the POST request and measure time
print("Sending test request...")
start_time = time.time()
try:
    # Set a 5s timeout and send the POST request
    response = requests.post(API_URL, json=payload, timeout=5)
    end_time = time.time()

    # 4. Check results
    response.raise_for_status()  # Raise exception for 4xx or 5xx errors
    result = response.json()

    print("-" * 50)
    print(f"Status Code: {response.status_code}")
    print(f"Response Latency: {(end_time - start_time):.3f} seconds")
    print(f"Latency Goal: < 3.000 seconds")
    print("-" * 50)
    print("Parsed Response Fields:")

    # Displaying the requested top-level fields
    print(f"  > ANSWER:")
    # Print the full answer, formatted
    answer = result.get('answer', 'N/A')
    print(f"    {answer}")
    print("-" * 50)

    # Displaying the CITATIONS field
    citations = result.get('citations', [])
    print(f"  > CITATIONS ({len(citations)} found):")

    if citations:
        # Loop through each citation object
        for i, citation in enumerate(citations):
            print(f"\n    --- Citation {i + 1} ---")

            # Display the nested fields
            content = citation.get('content', 'N/A')
            source_uri = citation.get('source_uri', 'N/A')

            # --- MODIFIED CODE BLOCK ---
            # 1. Prepare the content snippet, replacing newlines first.
            content_snippet = content[:200].replace('\n', ' ')

            print(f"      Source URI: {source_uri}")
            print(f"      Content (Snippet):")
            # 2. Use the prepared variable inside the f-string.
            print(f"        {content_snippet}...")
            # ---------------------------

    else:
        print("    No citations found.")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")