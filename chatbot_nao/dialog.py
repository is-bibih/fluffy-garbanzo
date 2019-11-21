import dialogflow
from google.api_core.exceptions import InvalidArgument

DIALOGFLOW_PROJECT_ID = 'brainvue-ufxcwl'
DIALOGFLOW_LANGUAGE_CODE = 'es'
SESSION_ID = 'uwu' # arbitrary string

session_client = dialogflow.SessionsClient()
session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)

def get_answer(text, show=False):
    """Return a response from the chatbot given a query."""
    text_input = dialogflow.types.TextInput(text=text, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise

    if show:
        print("Query text:", response.query_result.query_text)
        print("Detected intent:", response.query_result.intent.display_name)
        print("Detected intent confidence:", response.query_result.intent_detection_confidence)
        print("Fulfillment text:", response.query_result.fulfillment_text)

    return {"query": response.query_result.query_text,
            "intent": response.query_result.intent.display_name,
            "confidence": response.query_result.intent_detection_confidence,
            "fulfillment text": response.query_result.fulfillment_text}

def have_text_conversation():
    text = input("habla con pancho el robot :)\ncuando quieras salir, escribe 'salir'\n\n")
    while text != 'salir':
        reply = get_answer(text)
        if reply["fulfillment text"]:
            print("\n" + reply["fulfillment text"] + "\n")
        else:
            print("*intent detectado: {}*\n".format(reply["intent"]))
        text = input()
    print("\nbai")
