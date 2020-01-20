import requests
import time
import logger as lg

logger = lg.logging.getLogger(name='DialogFlow')
logger.setLevel(lg.logging.DEBUG)

_base_url = "https://api.dialogflow.com/v1/"
_version = "?v=20150910"
_instance = [
    {
        "client_token": "e0c64091d526433ab40937932719fc96",
        "developer_token": "2e9c8dad13874d498c31411739afa65d",
        "project_id": "aidmecompetitor-gwlrlf",
        "fallback_intent_id": "46038444-3cf4-4d63-a205-0186726af82d",
        "creation_counter": 1
    },
    {
        "client_token": "8af2ad1501214bb7be9846a043716bea",
        "developer_token": "2d1cb49a4821473ab3aba2e907a523c4",
        "project_id": "aidmecompetitor2-rfowol",
        "fallback_intent_id": "13dd6e14-f2b4-4826-ade2-29d23ce83005",
        "creation_counter": 1
    }
]

long_sleep = 10
short_sleep = 5


def detect_intent(instance, sentence, recall=False):
    url = _base_url + "query" + _version
    response = requests.post(url,
                             headers={"Authorization": "Bearer " + _instance[instance]["client_token"]},
                             json={"sessionId": "123456", "query": sentence, "lang": "en"})
    response_json = response.json()
    if response_json['status']['code'] == 200:
        intent_id = response_json["result"]["metadata"]["intentId"]
        if intent_id == _instance[instance]["fallback_intent_id"]:
            return None
        else:
            return {"intent_id": intent_id, "parameters": response_json["result"]["parameters"]}
    elif response_json['status']['code'] == 412:
        if recall:
            return None
        else:
            logger.info(response_json)
            time.sleep(10)
            return detect_intent(instance, sentence, recall=True)
    elif response_json['status']['code'] == 429:
        if recall:
            return None
        else:
            logger.info(response_json)
            time.sleep(61)
            return detect_intent(instance, sentence, recall=True)
    else:
        raise


def _get_intent_ids(instance, recall=False):
    url = _base_url + "intents" + _version
    response = requests.get(url, headers={"Authorization": "Bearer " + _instance[instance]["developer_token"]})
    intent_ids = []
    if response.status_code == 429:
        logger.info(response)
        time.sleep(long_sleep)
        return _get_intent_ids(instance)
    for intent in response.json():
        if intent["id"] != _instance[instance]["fallback_intent_id"]:
            intent_ids.append(intent["id"])
    return intent_ids


def _get_intent(instance, intent_id):
    url = _base_url + "intents" + "/" + str(intent_id) + _version
    response = requests.get(url, headers={"Authorization": "Bearer " + _instance[instance]["developer_token"]})
    if response.status_code == 429:
        logger.info(response)
        time.sleep(long_sleep)
        return _get_intent(instance, intent_id)
    return response.json()


def update_intent(instance, intent_id, *examples):
    """
    *examples = [{'pattern':'my pattern', variables:{'var_name':'value'}}]
    """
    url = _base_url + "intents" + _version + "&lang=en"
    intent = _get_intent(instance, intent_id)
    formatted_examples = _format_examples(examples)
    _fill_intent(intent, formatted_examples["sentences"])
    response = requests.put(url,
                 headers={"Authorization": "Bearer " + _instance[instance]["developer_token"]},
                 json=intent)
    if response.status_code == 429:
        logger.info(response)
        time.sleep(long_sleep)
        update_intent(instance, intent_id, *examples)


def create_intent(instance, *examples):
    """
    *examples = [{'pattern':'my pattern', variables:{'var_name':'value'}}]
    """
    url = _base_url + "intents" + _version + "&lang=en"
    formatted_examples = _format_examples(examples)
    intent = _create_base_intent(instance, formatted_examples["variables"])
    _fill_intent(intent, formatted_examples["sentences"])
    response = requests.post(url,
                             headers={"Authorization": "Bearer " + _instance[instance]["developer_token"]},
                             json=intent)
    if response.status_code == 429:
        logger.info(response)
        time.sleep(long_sleep)
        return create_intent(instance, *examples)
    return response.json()  # ["status"]


def _format_examples(examples):
    formatted_examples = {"variables": [], "sentences": []}
    for var in examples[0]["variables"]:
        formatted_examples["variables"].append(var[2:])
    for example in examples:
        formatted_sentence = []
        pattern = example["pattern"]
        current_part = ""
        in_var = False
        for c in pattern:
            if in_var:
                if c in (" ", "?", ".", ",", "!"):
                    formatted_sentence.append({'text': example["variables"][current_part],
                                               'alias': current_part[2:],
                                               'meta': '@sys.any',
                                               'userDefined': True})
                    in_var = False
                    current_part = ""
                current_part = current_part + c
            else:
                if c == "_":
                    in_var = True
                    if len(current_part) > 0:
                        formatted_sentence.append({'text': current_part, 'userDefined': False})
                        current_part = ""
                current_part = current_part + c
        if len(current_part) > 0:
            if in_var:
                formatted_sentence.append({'text': example["variables"][current_part],
                                           'alias': current_part[2:],
                                           'meta': '@sys.any',
                                           'userDefined': True})
            else:
                formatted_sentence.append({'text': current_part, 'userDefined': False})
        formatted_examples["sentences"].append(formatted_sentence)
    return formatted_examples


def _create_base_intent(instance, var_names):
    global _instance
    intent = {'name': 'intent' + str(_instance[instance]["creation_counter"]),
              'auto': True,
              'contexts': [],
              'responses': [
                  {'resetContexts': False,
                   'action': 'action' + str(_instance[instance]["creation_counter"]),
                   'affectedContexts': [],
                   'parameters': [],
                   'messages': [{'type': 0, 'condition': '', 'speech': []}],
                   'defaultResponsePlatforms': {},
                   'speech': []}],
              'priority': 500000,
              'webhookUsed': False,
              'webhookForSlotFilling': False,
              'fallbackIntent': False,
              'events': [],
              'userSays': [],
              'followUpIntents': [],
              'liveAgentHandoff': False,
              'endInteraction': False,
              'conditionalResponses': [],
              'condition': '',
              'conditionalFollowupEvents': [],
              'templates': []}
    for v in var_names:
        param = {'required': True,
                 'dataType': '@sys.any',
                 'name': v,
                 'value': '$' + v,
                 'promptMessages': [],
                 'noMatchPromptMessages': [],
                 'noInputPromptMessages': [],
                 'outputDialogContexts': [],
                 'isList': False}
        intent["responses"][0]["parameters"].append(param)
    _instance[instance]["creation_counter"] += 1
    return intent


def _fill_intent(intent, formatted_sentences):
    for formatted_sentence in formatted_sentences:
        intent["userSays"].append({'data': formatted_sentence,
                                   'isTemplate': False,
                                   'count': 0,
                                   'updated': 0,
                                   'isAuto': False})


def reset(instance):
    for intent_id in _get_intent_ids(instance):
        time.sleep(0.5)
        url = _base_url + "intents" + "/" + intent_id + _version
        response = requests.delete(url, headers={"Authorization": "Bearer " + _instance[instance]["developer_token"]})
        if response.status_code == 429:
            logger.info(response)
            time.sleep(long_sleep)
            reset(instance)


__save = {'name': 'myintentname',
          'auto': True,
          'contexts': [],
          'responses': [
              {'resetContexts': False,
               'action': 'myintentactionname',
               'affectedContexts': [],
               'parameters': [
                   {'id': '0e8f2b46-a390-4c11-b98b-505624f99cc7',
                    'required': True,
                    'dataType': '@sys.any',
                    'name': 'pers1',
                    'value': '$pers1',
                    'promptMessages': [],
                    'noMatchPromptMessages': [],
                    'noInputPromptMessages': [],
                    'outputDialogContexts': [],
                    'isList': False},
                   {'id': '8b313d02-8229-4baf-b34a-c9792f4a8a07',
                    'required': True,
                    'dataType': '@sys.any',
                    'name': 'pers2',
                    'value': '$pers2',
                    'promptMessages': [],
                    'noMatchPromptMessages': [],
                    'noInputPromptMessages': [],
                    'outputDialogContexts': [],
                    'isList': False}],
               'messages': [{'type': 0, 'condition': '', 'speech': []}],
               'defaultResponsePlatforms': {},
               'speech': []}],
          'priority': 500000,
          'webhookUsed': False,
          'webhookForSlotFilling': False,
          'fallbackIntent': False,
          'events': [],
          'userSays': [
              {'id': '889365fa-e27c-4c42-922b-f32a33c4ad7f',
               'data': [
                   {'text': 'Suzy', 'alias': 'pers1', 'meta': '@sys.any', 'userDefined': True},
                   {'text': ' kill ', 'userDefined': False},
                   {'text': 'Alice', 'alias': 'pers2', 'meta': '@sys.any', 'userDefined': True}]
                  , 'isTemplate': False,
               'count': 0,
               'updated': 0,
               'isAuto': False},
              {'id': '036d4a3e-12d9-40c1-9191-37997d350738',
               'data': [
                   {'text': 'Jo', 'alias': 'pers1', 'meta': '@sys.any', 'userDefined': False},
                   {'text': ' kill ', 'userDefined': False},
                   {'text': 'Bill', 'alias': 'pers2', 'meta': '@sys.any', 'userDefined': True}],
               'isTemplate': False,
               'count': 0,
               'updated': 0,
               'isAuto': False}],
          'followUpIntents': [],
          'liveAgentHandoff': False,
          'endInteraction': False,
          'conditionalResponses': [],
          'condition': '',
          'conditionalFollowupEvents': [],
          'templates': []}


class DialogFlowAgent:
    def __init__(self, instance, name, train_freq, initial_train_corpus=[], **kwargs):
        self.instance = instance
        self.name = name

        # params
        self.train_freq = train_freq
        self.since_last_train = 0

        reset(instance)
        self.intent_to_dialogflow_id = {}
        self.pattern_memory = set()
        self.dialog_flow_memory = []
        for i in initial_train_corpus:
            self.update_agent(i['intent'], i['pattern'], i['args'])

        self.update_model()

    def update_agent(self, intent, pattern, args):
        self.dialog_flow_memory.append(
            {'intent': intent, 'pattern': pattern, 'variables': args})

    def update_model(self):
        logger.info('Update Dialogflow')
        for i in self.dialog_flow_memory:
            time.sleep(0.5)
            intent = i.pop('intent')
            self.pattern_memory.add(i['pattern'])
            if intent not in self.intent_to_dialogflow_id.keys():
                response = create_intent(self.instance, i)
                self.intent_to_dialogflow_id[intent] = response['id']
            else:
                update_intent(self.instance, self.intent_to_dialogflow_id[intent], i)

        self.dialog_flow_memory = []
        self.since_last_train = 0
        time.sleep(5)  # Slow down calls

    @staticmethod
    def get_metrics(sentence_dict, new_intent, new_pattern, dialogflow_intent=None, dialogflow_pattern=None,
                    dialogflow_duration=0):
        metric = {
            'sentence': sentence_dict['sentence'],  # string
            'number_of_variables': sentence_dict["number_of_variables"],
            'number_of_domains': sentence_dict["number_of_domains"],
            'complexity': sentence_dict["complexity"],
            'new_intent': new_intent,
            'new_pattern': new_pattern,

            # duration
            'dialogflow_duration': dialogflow_duration,

            # dialogflow
            'dialogflow_intent': dialogflow_intent,
            'dialogflow_pattern': dialogflow_pattern,

        }
        return metric

    def is_new_intent_and_pattern(self, intent, pattern):  # TODO check
        is_new_intent = intent not in self.intent_to_dialogflow_id.keys()
        is_new_pattern = pattern not in self.pattern_memory
        return is_new_intent, is_new_pattern

    def dialogflow_prediction(self, sentence_dict):
        true_intent = sentence_dict["intent"]
        true_args = sentence_dict["args"]
        sentence = sentence_dict["sentence"]
        response = detect_intent(self.instance, sentence)
        if response is None:
            dialogflow_intent_detected = False
            dialogflow_pattern_detected = False
        else:
            try:
                dialogflow_intent_detected = (response['intent_id'] == self.intent_to_dialogflow_id[true_intent])
                vars = {}
                for k, v in response['parameters'].items():
                    vars['__' + k] = v
                dialogflow_pattern_detected = (vars == true_args)
            except KeyError:
                # When true_intent not in dialogflow base
                dialogflow_intent_detected = False
                dialogflow_pattern_detected = False
        return dialogflow_intent_detected, dialogflow_pattern_detected

    def run(self, sentence_dict):
        # first_sentence_dict = user_corpus.pop(0)
        # logger.info(f'Sentence number 1: {first_sentence_dict["sentence"]}')
        # self.update_agent(first_sentence_dict["intent"], first_sentence_dict["pattern"], first_sentence_dict["args"],
        #                   is_new_intent=True)
        # self.metrics.append(
        #     self.get_metrics(sentence_dict=first_sentence_dict, new_intent=True, new_pattern=True,
        #                      dialogflow_intent=False, dialogflow_pattern=False))
        #
        # i = 1
        # for sentence_dict in user_corpus:
        if self.since_last_train % self.train_freq == 0:
            self.update_model()

        true_intent = sentence_dict["intent"]
        true_pattern = sentence_dict["pattern"]
        true_args = sentence_dict["args"]

        is_new_intent, is_new_pattern = self.is_new_intent_and_pattern(true_intent, true_pattern)

        t = time.time()
        dialogflow_intent_detected, dialogflow_pattern_detected = self.dialogflow_prediction(sentence_dict)
        dialogflow_duration = round(time.time() - t, 3)

        sentence_metrics = self.get_metrics(sentence_dict=sentence_dict,

                                            new_intent=is_new_intent,
                                            new_pattern=is_new_pattern,

                                            dialogflow_duration=dialogflow_duration,
                                            dialogflow_intent=dialogflow_intent_detected,
                                            dialogflow_pattern=dialogflow_pattern_detected)

        self.update_agent(true_intent, true_pattern, true_args)
        self.since_last_train += 1
        return sentence_metrics


if __name__ == "__main__":
    import time

    # reset(0)
    # ex1 = {"pattern": "search __pers1 on __search1 then buy a ticket for his concert",
    #        'variables': {"__pers1": "Bob", "__search1": "wikipedia"}}
    # ex2 = {"pattern": "search __pers1 on __search1 then buy a ticket for his concert",
    #        'variables': {"__pers1": "Alice", "__search1": "wikipedia"}}

    print(len(_get_intent_ids(0)))
    # response = create_intent(0, ex1)
    # print(_get_intent_ids(0))
    # print(response)
    # id = response['id']
    # update_intent(0, id, ex2)
    # print(detect_intent(0, "search Dylan on google then buy a ticket for his concert"))
    # time.sleep(10)
    # print(detect_intent(0, "search Dylan on google then buy a ticket for his concert"))
    #
    # print(_get_intent_ids(0))
    # # update_intent(0, )
    # print(_get_intent(0, _get_intent_ids(0)[0]))
