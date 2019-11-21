"""Install the following requirements:
    dialogflow        0.5.1
    google-api-core   1.4.1

   Run in the session shell:
   export GOOGLE_APPLICATION_CREDENTIALS="/home/vivi/fluffy-garbanzo/chatbot_nao/brainvue-ufxcwl-620047b917a6.json"
"""
import dialogflow
from google.api_core.exceptions import InvalidArgument
DIALOGFLOW_PROJECT_ID = 'brainvue-ufxcwl'
DIALOGFLOW_LANGUAGE_CODE = 'es'
SESSION_ID = 'uwu' # arbitrary string

text_to_be_analyzed = "¿qué es brainvue?"
session_client = dialogflow.SessionsClient()
session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
text_input = dialogflow.types.TextInput(text=text_to_be_analyzed, language_code=DIALOGFLOW_LANGUAGE_CODE)
query_input = dialogflow.types.QueryInput(text=text_input)
try:
    response = session_client.detect_intent(session=session, query_input=query_input)
except InvalidArgument:
    raise
print("Query text:", response.query_result.query_text)
print("Detected intent:", response.query_result.intent.display_name)
print("Detected intent confidence:", response.query_result.intent_detection_confidence)
print("Fulfillment text:", response.query_result.fulfillment_text)
