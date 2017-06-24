import re
import language_check
import data_loader

# merge partial utterances belonging to the same MR into multi-sentence utterances
def merge_utterances(results, mrs, test_groups, num_variations):
    final_utterances = []
    merged_utterance = ''
    prev_group = -1

    for sent, cur_group in zip(results, test_groups):
        if cur_group != prev_group:
            if prev_group != -1:
                final_utterances.append(merged_utterance + '.')

            merged_utterance = relex_utterance(sent, mrs[cur_group // num_variations])
            prev_group = cur_group
        else:
            merged_utterance += '. ' + relex_utterance(sent, mrs[cur_group // num_variations], replace_name=True)
    
    final_utterances.append(merged_utterance + '.')

    return final_utterances


def relex_utterance(utterance, mr, replace_name=False):
    # parse the slot-value pairs from the MR
    slots = {}
    for slot_value in mr.split(','):
        sep_idx = slot_value.find('[')
        # parse the slot and the value
        slot = slot_value[:sep_idx].strip()
        value = slot_value[sep_idx + 1:-1].strip()
        slots[slot] = value
    
    # identify all value placeholders
    matches = re.findall(r'&slot_val_.*?&', utterance)
    
    # replace the value placeholders with the corresponding values from the MR
    fail_flags = []
    for match in matches:
        slot = match.split('_')
        slot = slot[-1].rstrip('&')
        if slot in list(slots.keys()):
            if slot == 'name' and replace_name:
                new_val = 'It'
            else:
                new_val = slots[slot]
            utterance = utterance.replace(match, new_val)
        else:
            fail_flags.append(slot)

    # capitalize the first letter of each sentence
    utterance = utterance[0].upper() + utterance[1:]
    sent_end = utterance.find(r'-PERIOD-')
    while sent_end >= 0:
        next_sent_beg = sent_end + 2
        if next_sent_beg < len(utterance):
            utterance = utterance[:next_sent_beg] + utterance[next_sent_beg].upper() + utterance[next_sent_beg + 1:]
        
        sent_end = utterance.find(r'-PERIOD-', next_sent_beg)
    
    # replace the period placeholders
    utterance = utterance.replace(r' -PERIOD-', '.')


    if len(fail_flags) > 0:
        print('When relexing, the following slots could not be handled by the MR: ' + str(fail_flags))
        print(utterance)
        print(mr)

    return utterance


def combo_print(small_pred, large_pred, num_permutes):
    x = 0
    y = 0
    base = max(int(len(small_pred) * .1), 1)
    benchmarks = [base * i for i in range(1, 11)]
    new_pred = []
    while x < len(small_pred):
        if x in benchmarks:
            curr_state = x / base
            print("Depermute processing is " + str(10 * curr_state) + "% done.")

        for i in range(0, num_permutes):
            new_pred.append(large_pred[x*num_permutes+i])
        new_pred.append('\033[1m' + small_pred[x] + '\033[0m')
        x += 1
    return new_pred


def depermute_input(mrs, sents, predictions, num_permutes):
    new_mr = []
    new_sent = []
    new_pred = []
    x = 0
    tool = language_check.LanguageTool('en-UK')

    base = max(int(len(predictions) * .1), 1)
    benchmarks = [base*i for i in range(1, 11)]
    
    while x < len(predictions):
        if x in benchmarks:
            curr_state = x / base
            print("Depermute processing is " + str(10 * curr_state) + "% done.")

        scores = {}
        for i in range(0, num_permutes):
            scores[x + i] = score_output(mrs[x // num_permutes], sents[x // num_permutes], predictions[x + i], tool, correction=False)
        
        top_score = max(scores.keys(), key=(lambda key: scores[key]))
        new_mr.append(mrs[top_score // num_permutes])
        new_sent.append(sents[top_score // num_permutes])
        new_pred.append(predictions[top_score])
        x += num_permutes

    return new_mr, new_sent, new_pred

def correct(mrs, pred):
    print("Correcting outputs:")
    new_pred = []
    base = max(int(len(pred) * .1),1)
    benchmarks = [base * i for i in range(1, 11)]

    tool = language_check.LanguageTool('en-UK')
    for x, p in enumerate(pred):
        if x in benchmarks:
            curr_state = x / base
            print("Correct processing is " + str(10 * curr_state) + "% done.")

        s1, c1 = score_grammar_spelling(mrs[x], p, tool, True)
        s1, c1 = score_known_errors(c1, True)
        new_pred.append(c1)
    return new_pred


def score_output(mr, sent, pred, tool=None, correction=True):
    #todo do we want to allow score to go negative?
    score = 0
    score += score_informativeness(mr, pred)
    score -= score_grammar_spelling(mr, pred, tool)
    score -= score_known_errors(pred)
    return score


def score_informativeness(mr, pred):
    score = 0
    mrs = mr.split(',')
    for slot_value in mrs:
        sep_idx = slot_value.find('[')
        # slot = slot_value[:sep_idx].strip()
        value = slot_value[sep_idx + 1:-1].strip()
        value = value.lower()

        # score += sent.count(value)
        #this won't account for duplicates
        if value in pred.lower():
            score += 1
    #normalize score with possible mrs
    score = score/len(mrs)
    return score


def score_grammar_spelling(mr, pred, tool=None, correct=False):
    #FIXME this will lower case text such as Travellers which it doesn't see as a NNP, is this fine?
    #FIXME do we want to apply the corrections or just pick from the best result?
    pred = data_loader.delex_data([mr], [pred], update_data_source=True, specific_slots=None, split=False)
    if tool is None:
        tool = language_check.LanguageTool('en-UK')
    matches = tool.check(pred)
    score = min(len(matches)/len(pred.split()), 1)
    if correct:
        x = 0
        while True:
            new_pred = tool.correct(pred)
            if pred == new_pred or x == 5:
                break
            pred = new_pred
            x += 1
        pred = relex_utterance(pred, mr)
        return score, pred
    return score


def score_known_errors(pred, correct=False):
    pred_split = pred.split()
    score = 0
    temp_score = 0
    var_to_reduce = []
    prev = None
    for ps in pred_split:
        #accounts for a wierd case of like 5 5 5 5 5
        if len(ps) == 1 and ps in ["0","1","2","3","4","5","6","7","8","9"] and (prev == ps or prev is None):
            temp_score += 1
            prev = ps
        else:
            if temp_score > 1:
                score += temp_score
                var_to_reduce.append((prev,temp_score),)
            temp_score = 0
            if ps in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                prev = ps
                temp_score += 1
    if temp_score > 1:
        score += temp_score
        var_to_reduce.append((prev, temp_score), )
    #TODO untested
    if correct:
        for var in var_to_reduce:
            v, num_v = var
            string_to_kill = " ".join([v]*num_v)
            pred = pred.replace(string_to_kill, " "+v)
        return score/len(pred_split), pred

    return score/len(pred_split)
