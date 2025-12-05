import streamlit as st
import requests
import json
import time  # For showing the latency in the UI

# --- CONFIGURATION ---
# Replace this with the INVOKE URL of your deployed API Gateway /query stage
API_GATEWAY_URL = "https://oqn6cgfx29.execute-api.ap-southeast-2.amazonaws.com/prod/query"

# --- UI SETUP ---
st.set_page_config(
    page_title="MedAI RAG QA System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🩺 MedAI Retrieval-Augmented Generation (RAG)")
st.caption("Powered by AWS API Gateway, Lambda, and Bedrock Knowledge Base.")
st.markdown("---")


# --- CORE API CALL FUNCTION ---
def get_rag_answer(prompt):
    """Sends the user prompt to the API Gateway and handles the response."""

    payload = {"prompt": prompt}

    start_time = time.time()

    try:
        response = requests.post(API_GATEWAY_URL, json=payload)
        latency = time.time() - start_time

        if response.status_code == 200:
            # 1. Parse the initial API Gateway JSON response
            api_response = response.json()

            # 2. Check for the 'body' key (where Lambda output is stored as a string)
            if 'body' in api_response and isinstance(api_response['body'], str):

                # 3. Parse the nested JSON string to get the final result dictionary
                rag_result = json.loads(api_response['body'])
            else:
                # Fallback for direct Lambda invocation where 'body' isn't used
                rag_result = api_response

            return rag_result, latency

        else:
            # Handle non-200 API errors (e.g., 500 from Lambda error)
            error_message = f"API Error ({response.status_code}): {response.text}"
            return {"error": error_message}, latency

    except requests.exceptions.RequestException as e:
        latency = time.time() - start_time
        return {"error": f"Connection Error: {e}"}, latency


# --- STREAMLIT UI LAYOUT ---

# 1. User Input Area
user_query = st.text_area(
    "Enter your medical question:",
    placeholder="e.g., What are the standard treatment protocols for Type 2 Diabetes according to your knowledge base?",
    height=100
)

# 2. Submit Button
if st.button("Get RAG Answer", type="primary", use_container_width=True):
    if not user_query:
        st.error("Please enter a question to get an answer.")
    else:
        # Show a spinner while processing
        with st.spinner(
                "Processing request through API Gateway... (This takes about 5.5 seconds)"):

            # Call the function to hit the API
            result, latency = get_rag_answer(user_query)

            # Display the results
            if "error" in result:
                st.error(f"Execution Failed: {result['error']}")

            else:
                # Structure the output using columns
                col_latency, col_certainty = st.columns([1, 1])

                # Column 1: Latency Display (for Phase 7 Reporting)
                col_latency.metric(
                    label="End-to-End Latency",
                    value=f"{latency:.3f} s",
                    delta_color="off"
                )

                # Column 2: Placeholder for Short Answer/Certainty
                col_certainty.metric(
                    label="Certainty / Short Answer",
                    value=f"{result.get('certainty', 'N/A')}",
                    delta=result.get('short_answer', 'Answer N/A'),
                    # Display short answer in delta
                    delta_color="off"
                )

                st.markdown("---")

                # Long Answer Display
                st.subheader("💡 RAG Answer")
                st.markdown(result.get("answer",
                                       "No answer was returned by the model."),
                            unsafe_allow_html=True)

                st.markdown("---")

                # Citations Display (using an Expander for cleanliness)
                st.subheader("🔗 Citations / Source Documents")
                citations = result.get("citations", [])

                if citations:
                    with st.expander(
                            f"View {len(citations)} Retrieved Sources"):
                        for i, ref in enumerate(citations):
                            uri = ref.get('source_uri', 'N/A: URI Missing')
                            st.markdown(f"**Source {i + 1}:**")
                            st.code(f"Source URI: {uri}", language="markdown")
                            # **Ensure this key matches what your Lambda outputs!**
                            st.markdown(
                                f"**Retrieved Text:** {ref.get('content', 'No text found.')}")
                            st.divider()
                else:
                    st.warning("No citations were returned for this query.")

                # For debugging: show the full raw result
                with st.expander("🐞 View Full Raw Result (for Bugtesting)"):
                    st.json(result)

# --- RUNNING THE APP ---
# To run this script locally, use the command: streamlit run app.py