import numpy

from UserGrammar import corpus_data

domains = []
for k in corpus_data.data:
    domains.extend(k["domains"])
domains = list(set(domains))

standard_domain_distribution = {k: 1. / len(domains) for k in domains}


def generate_user_corpus(number_of_sentences, train_used_patterns, proportion_of_novelty,
                         domain_distribution=standard_domain_distribution):
    unused_patterns = _get_all_possible_patterns()
    for p in train_used_patterns:
        unused_patterns.remove(p)
    n_sentences_in_train = round((1 - proportion_of_novelty) * number_of_sentences)
    n_sentences_not_in_train = number_of_sentences - n_sentences_in_train
    sentences = _generate_corpus(n_sentences_not_in_train, unused_patterns, domain_distribution)
    sentences.extend(_generate_corpus(n_sentences_in_train, train_used_patterns, domain_distribution))
    numpy.random.shuffle(sentences)
    sentences.sort(key=lambda s: s["complexity"])  # maybe some use case will only consider number_of_variables
    return sentences


def generate_train_corpus(corpus_size, n_sentence, domain_distribution=standard_domain_distribution,
                          filtered_pattern=[]):
    corpus = []
    used_patterns = []
    sentence_corpus = _generate_corpus(n_sentence, filtered_pattern=filtered_pattern,
                                       domain_distribution=domain_distribution)
    for i in range(corpus_size):
        option1 = sentence_corpus[numpy.random.randint(0, len(sentence_corpus))]
        option2 = sentence_corpus[numpy.random.randint(0, len(sentence_corpus))]
        score = 0
        if option1["intent"] == option2["intent"]:
            score = 1
        used_patterns.append(option1["pattern"])
        used_patterns.append(option2["pattern"])
        corpus.append([option1["sentence"], option2["sentence"], score])
    return {"used_patterns": list(set(used_patterns)), "corpus": corpus}


def _generate_corpus(number_of_sentences, filtered_pattern=[], domain_distribution=standard_domain_distribution):
    sentences = []
    corpus_data_per_domain = _get_corpus_data_per_domain(filtered_pattern)
    number_of_sentences_per_domain = {}
    remaining_number_of_sentences = number_of_sentences
    for domain, proportion in domain_distribution.items():
        n = round(proportion * number_of_sentences)
        number_of_sentences_per_domain[domain] = min(n, remaining_number_of_sentences)
        remaining_number_of_sentences = remaining_number_of_sentences - n
    for domain, number in number_of_sentences_per_domain.items():
        for i in range(number):
            domain_data = corpus_data_per_domain[domain]
            if not domain_data:
                raise LookupError
            intent = domain_data[numpy.random.randint(0, len(domain_data))]
            intent_id = intent["intent_id"]
            patterns = intent["patterns"]
            pattern_data_index = numpy.random.randint(0, len(patterns))
            pattern_data = patterns[pattern_data_index]
            sentence = pattern_data["pattern"]
            number_of_domains = pattern_data["number_of_domains"]
            variables = _find_variables(sentence)
            if "possible_bindings" not in pattern_data:
                pattern_data["possible_bindings"] = _compute_possible_bindings(variables)
            variable_binding = {}
            if pattern_data["possible_bindings"]:
                i = numpy.random.randint(0, len(pattern_data["possible_bindings"]))
                variable_binding = pattern_data["possible_bindings"].pop(i)
            if not pattern_data["possible_bindings"]:
                del patterns[pattern_data_index]
                if not intent["patterns"]:
                    for list_of_intent in corpus_data_per_domain.values():
                        if intent in list_of_intent:
                            list_of_intent.remove(intent)
            sentence = [variable_binding[w] if w in variables else w for w in sentence.split(' ')]
            sentence = ' '.join(sentence)
            sentences.append({"intent": intent_id,
                              "pattern": pattern_data['pattern'],
                              'args': variable_binding,
                              "sentence": sentence,
                              "number_of_variables": len(variables),
                              "number_of_domains": number_of_domains,
                              "complexity": number_of_domains * 10 + len(variables),
                              "possible_patterns": _get_possible_patterns(intent_id),
                              "possible_sentences": _get_possible_sentences(intent_id, variable_binding)})
    numpy.random.shuffle(sentences)
    sentences.sort(key=lambda s: s["complexity"])  # maybe some use case will only consider number_of_variables
    return sentences


def _find_variables(pattern):
    variable_prefix = corpus_data.possible_values.keys()
    variables = []
    for w in pattern.split(' '):
        for prefix in variable_prefix:
            if prefix in w:
                variables.append(w)
    return list(set(variables))


def _compute_possible_bindings(variables):
    possible_bindings = []
    _compute_possible_bindings_recur(variables, {}, possible_bindings)
    return possible_bindings


def _compute_possible_bindings_recur(variables, possible_binding, possible_bindings):
    if not variables:
        if possible_binding:
            possible_bindings.append(possible_binding)
        return
    new_variables = variables[:]
    variable = new_variables.pop(0)
    var_domain = []
    for var_type, value_domain in corpus_data.possible_values.items():
        if variable.startswith(var_type):
            var_domain = value_domain
            break
    if not var_domain:
        raise RuntimeError
    for value in var_domain:
        new_possible_binding = dict(possible_binding)
        new_possible_binding[variable] = value
        _compute_possible_bindings_recur(new_variables, new_possible_binding, possible_bindings)


def _get_possible_patterns(intent_id):
    return corpus_data.data[intent_id]["patterns"][:]


def _get_all_possible_patterns():
    patterns = []
    for intent in corpus_data.data:
        patterns.extend(intent["patterns"][:])
    return patterns


def _get_possible_sentences(intent_id, variable_binding):
    sentences = []
    for sentence in _get_possible_patterns(intent_id):
        for variable, value in variable_binding.items():
            sentence = sentence.replace(variable, value)
        sentences.append(sentence)
    return sentences


def _get_corpus_data_per_domain(filtered_pattern):
    corpus_data_per_domain = {k: [] for k in domains}
    for i in range(len(corpus_data.data)):
        intent = {"intent_id": i, "patterns": []}
        for pattern in corpus_data.data[i]["patterns"]:
            if pattern not in filtered_pattern:
                intent["patterns"].append(
                    {"pattern": pattern, "number_of_domains": len(corpus_data.data[i]["domains"])})
        if intent["patterns"]:
            for domain in corpus_data.data[i]["domains"]:
                corpus_data_per_domain[domain].append(intent)
    return corpus_data_per_domain


if __name__ == '__main__':
    print(domains, standard_domain_distribution)
    # train_data = generate_train_corpus(corpus_size=20, n_sentence=5)
    # train_corpus, train_patterns = train_data['corpus'], train_data['used_patterns']
    # user_corpus = generate_user_corpus(number_of_sentences=30,
    #                                    train_used_patterns=train_patterns,
    #                                    proportion_of_novelty=0.25)

    train_corpus = generate_train_corpus(4000, 150)
    print(len(train_corpus['used_patterns']))
    test_corpus = generate_train_corpus(2000, 100, filtered_pattern=train_corpus['used_patterns'])
