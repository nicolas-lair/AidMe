import re

from UserGrammar.corpus import _find_variables


def example(pattern="Send an email to __var1 about __var2 and __var1",
            sentence="Please, write an email to bill about coffee and bill", max_words_per_variable: int = 1):
    for s, p in generate(pattern, sentence, max_words_per_variable=max_words_per_variable).items():
        print(s + " <--> " + p)


def generate(pattern: str, sentence: str, max_words_per_variable: int = 1) -> dict:
    """
    Generate all the possible instances of a pattern given a sentence.

    Take a pattern such as "Send an email to __var1 about __var2", where words starting with the specified prefix are
    variables, a sentence like "Please, write an email to clem@dede.com about coffee", and the maximum number of words
    per variables. It returns a dictionary that maps instantiated patterns (with all the possible arrangements of the
     words in the sentence) with a new pattern derived from the sentence. Here is an expected result:

    * { 'Send an email to Please about write' : ' __var1, __var2 an email to clem@dede.com about coffee',
        'Send an email to Please about an': '__var1, write __var2 email to clem@dede.com about coffee',
        ...
        }

    """
    pattern = _normalize(pattern)
    sentence = _normalize(sentence)
    variables = _find_variables(pattern)

    if len(variables) == 0:
        return {pattern: sentence}
    else:
        words = _to_words(sentence)
        possible_values = _compute_possible_values(words, max_words_per_variable)
        arrangements = _generate_arrangement(variables, possible_values, len(words))
        return _format_result(pattern, words, arrangements)


def _format_result(pattern, words, arrangements):
    result = {}
    for arrangement in arrangements:
        generated_pattern = ""
        i = 0
        while i < len(words):
            start_index = i
            replacement = None
            for (variable, possible_value) in arrangement.items():
                if possible_value['start_index'] == start_index:
                    replacement = {'variable': variable, 'index': possible_value['end_index']}
                    break
            if replacement is not None:
                generated_pattern = generated_pattern + replacement['variable']
                i = replacement['index']
            else:
                generated_pattern = generated_pattern + words[i]

            i = i + 1

        generated_sentence = pattern
        for variable in arrangement:
            value = ""
            j = arrangement[variable]['start_index']
            while j <= arrangement[variable]['end_index']:
                value = value + words[j]
                j = j + 1
            generated_sentence = generated_sentence.replace(variable, value)

        result[generated_sentence] = generated_pattern

    return result


def _generate_arrangement(variables, possible_values, max_size):
    arrangements = []
    _generate_combinations(possible_values[:],
                           [None] * len(variables),
                           0,
                           len(possible_values) - 1,
                           0,
                           len(variables),
                           variables[:],
                           arrangements,
                           max_size)
    return arrangements


def _generate_combinations(possible_values, temp_combination, start, end, index, k, variables, result, max_size):
    if index == k:
        if _check_not_overlap(temp_combination, max_size):
            _generate_permutations(0, len(variables) - 1, variables, temp_combination[:], result)
    else:
        i = start
        while i <= end and end - i + 1 >= k - index:
            temp_combination[index] = possible_values[i]
            _generate_combinations(possible_values,
                                   temp_combination,
                                   i + 1,
                                   end,
                                   index + 1,
                                   k,
                                   variables,
                                   result,
                                   max_size)
            i = i + 1


def _generate_permutations(i, j, variables, temp_permutation, result):
    if i == j:
        permutation = {}
        k = 0
        while k < len(variables):
            permutation[variables[k]] = temp_permutation[k]
            k = k + 1
        result.append(permutation)
    else:
        k = i
        while k <= j:
            _swap(temp_permutation, i, k)
            _generate_permutations(i + 1, j, variables, temp_permutation, result)
            k = k + 1


def _swap(table, i, j):
    tmp = table[i]
    table[i] = table[j]
    table[j] = tmp


def _check_not_overlap(table, max_size):
    result = True
    a = [False] * max_size
    i = 0
    while i < len(table) and result:
        j = table[i]['start_index']
        while j <= table[i]['end_index']:
            if j != table[i]['end_index'] or table[i]['start_index'] != table[i]['end_index']:
                if a[j]:
                    result = False
                else:
                    a[j] = True
            j = j + 1
        i = i + 1
    return result;


def _compute_possible_values(words, max_words_per_variable):
    possible_values = []
    for group in _compute_possible_values_groups(words):
        possible_values = possible_values + group[:]
        max_words = min(max_words_per_variable, len(group))
        m = 2
        while m <= max_words:
            i = 0
            while i + m - 1 < len(group):
                start = group[i]['start_index']
                end = group[i + m - 1]['end_index']
                possible_values.append({'start_index': start, 'end_index': end})
                i = i + 1
            m = m + 1
    return possible_values


def _compute_possible_values_groups(words):
    possible_values_groups = []
    group = []
    start = 0
    end = 0
    s = 0
    i = 0
    while i <= len(words):
        if i == len(words) or words[i] == " ":
            l = []
            j = s
            while j < i:
                l.append(words[j])
                j = j + 1
            start = s + _cleanup_begin(l)
            l_size = len(l)
            end = start + l_size - _cleanup_end(l) - 1
            if len(l) > 0:
                value = ""
                k = start
                while k <= end:
                    value = value + words[k]
                    k = k + 1
                group.append({'start_index': start, 'end_index': end})
            s = i
            if len(l) != l_size:
                possible_values_groups.append(group)
                group = []
        i = i + 1
    if len(group) > 0:
        possible_values_groups.append(group)
    return possible_values_groups


def _cleanup_end(l):
    end = False
    count = 0
    while len(l) > 0 and not end:
        e = l[len(l) - 1]
        if e == " " or (len(e) == 1 and not e.isalnum()):
            count = count + 1
            del l[len(l) - 1]
        else:
            end = True
    return count


def _cleanup_begin(l):
    end = False
    count = 0
    while len(l) > 0 and not end:
        e = l[0]
        if e == " " or (len(e) == 1 and not e.isalnum()):
            count = count + 1
            del l[0]
        else:
            end = True
    return count


# def _find_variables(pattern, variable_prefix):
#     variables = []
#     p = pattern
#     while len(p) > len(variable_prefix):
#         if p.startswith(variable_prefix):
#             variable = ""
#             found = False
#             n = 0
#             while n < len(p) and not found:
#                 if p[n] != " ":
#                     variable = variable + p[n]
#                 else:
#                     found = True
#                 n = n + 1
#             variables.append(variable)
#             p = p[len(variable):]
#         else:
#             p = p[1:]
#     return variables

# def _find_variables(pattern, variable_prefix):
#     variables = []
#     if isinstance(variable_prefix, str):
#         variable_prefix = [variable_prefix]
#     for w in pattern.split(' '):
#         for prefix in variable_prefix:
#             if prefix in w:
#                 variables.append(w)
#     return variables


def _normalize(sentence):
    return re.sub("\\s+", " ", sentence.strip())


def _to_words(sentence):
    words = []
    prev_alpha = False
    prev_digit = False
    current_word = ""
    for c in sentence:
        if c.isalpha():
            if prev_alpha:
                current_word = current_word + c
            else:
                if len(current_word) > 0:
                    words.append(current_word)
                current_word = "" + c
            prev_alpha = True
            prev_digit = False
        elif c.isdigit():
            if prev_digit:
                current_word = current_word + c
            else:
                if len(current_word) > 0:
                    words.append(current_word)
                current_word = "" + c
            prev_alpha = False
            prev_digit = True
        else:
            if len(current_word) > 0:
                words.append(current_word)
            current_word = "" + c
            prev_alpha = False
            prev_digit = False
    if len(current_word) > 0:
        words.append(current_word)
    return words


if __name__ == '__main__':
    p = 'please, refund my __num tickets for the __pers1 concert and buy __num other tickets for the __pers2 concert'
    s = 'please, refund my 5 tickets for the Bill concert and buy 5 other tickets for the Bob concert'
    example(pattern=p, sentence=s)
