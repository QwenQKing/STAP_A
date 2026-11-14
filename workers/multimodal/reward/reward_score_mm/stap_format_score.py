import re
import string


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    exactly_match = False
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            exactly_match = True
            break
    return exactly_match


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(prediction):

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, prediction, re.DOTALL)
    matches = list(match)

    if not matches:
        return None
    else:
        return matches[-1].group(1).strip()


def _tag_count_ok(response, tag, n) -> bool:
    return response.count(f'<{tag}>') == n and response.count(f'</{tag}>') == n


def _strict_shape(r: str, tags: list) -> bool:
    pat = r'^\s*' + r'\s*'.join(r'<%s>.*?</%s>' % (t, t) for t in tags) + r'\s*$'
    return re.match(pat, r.strip(), re.DOTALL) is not None


_STRATA = ('W0', 'W1', 'W2', 'W3', 'W4')


def _evaluate_ok(evaluate: str) -> bool:
    lines = [l.strip() for l in evaluate.strip().splitlines() if l.strip()]
    if len(lines) != len(_STRATA) + 1:
        return False
    for line, tag in zip(lines, _STRATA):
        m = re.match(r'^%s\s*[:：]\s*(\S.*)$' % tag, line)
        if not m:
            return False
    m = re.match(r'^Primary\s*[:：]\s*(W[0-4])\b', lines[-1], re.IGNORECASE)
    return m is not None


def _check_round(assistant_block: str, kind: str) -> bool:
    r = assistant_block.strip()

    if kind == 'first':
        ok_count = (
            r.count('<reason>')   == 1 and r.count('</reason>')   == 1 and
            r.count('<generate>') == 1 and r.count('</generate>') == 1 and
            r.count('<evaluate>') == 0 and r.count('</evaluate>') == 0 and
            r.count('<complete>') == 0 and r.count('</complete>') == 0
        )
        if not ok_count:
            return False
        if not _strict_shape(r, ['reason', 'generate']):
            return False
        reason   = re.search(r'<reason>(.*?)</reason>',     r, re.DOTALL).group(1).strip()
        generate = re.search(r'<generate>(.*?)</generate>', r, re.DOTALL).group(1).strip()
        return bool(reason) and bool(generate)

    if kind == 'middle':
        ok_count = (
            r.count('<reason>')   == 1 and r.count('</reason>')   == 1 and
            r.count('<evaluate>') == 1 and r.count('</evaluate>') == 1 and
            r.count('<generate>') == 1 and r.count('</generate>') == 1 and
            r.count('<complete>') == 0 and r.count('</complete>') == 0
        )
        if not ok_count:
            return False
        if not _strict_shape(r, ['reason', 'evaluate', 'generate']):
            return False
        reason   = re.search(r'<reason>(.*?)</reason>',     r, re.DOTALL).group(1).strip()
        evaluate = re.search(r'<evaluate>(.*?)</evaluate>', r, re.DOTALL).group(1).strip()
        generate = re.search(r'<generate>(.*?)</generate>', r, re.DOTALL).group(1).strip()
        return bool(reason) and _evaluate_ok(evaluate) and bool(generate)

    if kind == 'last':
        ok_count = (
            r.count('<reason>')   == 1 and r.count('</reason>')   == 1 and
            r.count('<evaluate>') == 1 and r.count('</evaluate>') == 1 and
            r.count('<complete>') == 1 and r.count('</complete>') == 1 and
            r.count('<generate>') == 0 and r.count('</generate>') == 0
        )
        if not ok_count:
            return False
        if not _strict_shape(r, ['reason', 'evaluate', 'complete']):
            return False
        reason   = re.search(r'<reason>(.*?)</reason>',     r, re.DOTALL).group(1).strip()
        evaluate = re.search(r'<evaluate>(.*?)</evaluate>', r, re.DOTALL).group(1).strip()
        complete = re.search(r'<complete>(.*?)</complete>', r, re.DOTALL).group(1).strip()
        return bool(reason) and _evaluate_ok(evaluate) and complete == 'end'

    return False


ROUND_WEIGHTS = {'first': 0.3, 'middle': 0.2, 'last': 0.3}


def format_reward(input_string: list):
    n = len(input_string)
    if n == 0:
        return 0.0, 0
    weighted_sum = 0.0
    generate_count = 0
    for i, resp in enumerate(input_string):
        if '<generate>' in resp:
            generate_count += 1
        if i == 0:
            kind = 'first'
        elif i == n - 1:
            kind = 'last'
        else:
            kind = 'middle'
        if _check_round(resp, kind):
            weighted_sum += ROUND_WEIGHTS[kind]
    return min(weighted_sum, 1.0), generate_count


def compute_score(prediction: list, ground_truth: list, extra_info=None):
    search_penalty, format_penalty = 0.1, 0.1
    reward_mode = 'EM'
    if extra_info is not None and 'search_penalty' in extra_info:
        search_penalty = extra_info.get('search_penalty', 0.1)
    if extra_info is not None and 'format_penalty' in extra_info:
        format_penalty = extra_info.get('format_penalty', 0.1)
    if extra_info is not None and 'reward_mode' in extra_info:
        reward_mode = extra_info.get('reward_mode', 'EM')
        assert reward_mode in ['EM', 'SubEM'], f'reward mode {reward_mode} passed in extra_info but not recognized'

    assert len(prediction) > 0, "[Error Occured] Model Responses are empty!"
    answer = extract_solution(prediction=prediction[-1])

    score = 0
    if answer is not None:
        if reward_mode == "EM" and em_check(answer, ground_truth):
            score = 1
        elif reward_mode == 'SubEM' and subem_check(answer, ground_truth):
            score = 1

    format_score, generate_count = format_reward(prediction)

    if generate_count > 0 and score > 0.99:
        use_search_count_penalty = extra_info.get('use_search_count_penalty', False)
        if use_search_count_penalty:
            for _ in range(generate_count):
                score *= 1 - search_penalty
        else:
            score *= 1 - search_penalty

    return (1 - format_penalty) * score + format_penalty * format_score

