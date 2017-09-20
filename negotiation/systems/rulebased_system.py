from cocoa.systems.rulebased_system import RulebasedSystem as BaseRulebasedSystem
from sessions.rulebased_session import RulebasedSession

class RulebasedSystem(BaseRulebasedSystem):

    def __init__(self, lexicon, timed_session=False):
        super(RulebasedSystem, self).__init__(timed_session)
        self.lexicon = lexicon

    def _new_session(self, agent, kb, config=None):
        return RulebasedSession.get_session(agent, kb, self.lexicon, config)

class TemplateExtractor(object):
    from pygtrie import Trie
    from cocoa.core.dataset import read_examples
    from cocoa.core.entity import is_entity
    from core.scenario import Scenario
    from core.tokenizer import detokenize
    from analysis.dialogue import Utterance
    from analysis.speech_acts import SpeechActAnalyzer

    def __init__(self, price_tracker):
        self.price_tracker = price_tracker

    def tag_utterance(self, utterance):
        """Get speech acts of the utterance.
        """
        acts = []
        if SpeechActAnalyzer.is_question(utterance):
            acts.append('question')
        if utterance.prices:
            acts.append('price')
        if SpeechActAnalyzer.is_price(utterance):
            acts.append('vague-price')
        return acts

    def tag_utterances(self, utterances):
        """Tag utterances in the dialogue context.

        Args:
            utterances ([Utterance])

        """
        tagged_utterances = []
        N = len(utterances)
        prev_acts = []
        for i, utterance in enumerate(utterances):
            if utterance.action != 'message':
                continue
            acts = self.tag_utterance(utterance)
            tag = None
            if i == 0 and not 'price' in acts:
                tag = 'intro'
            elif i + 1 < N and utterances[i+1].event == 'offer' and not acts:
                tag = 'agree'
            elif (not 'price' in acts) and 'vague-price' in acts:
                tag = 'vague-price'
            elif 'price' in acts and 'price' in prev_acts:
                tag = 'counter-price'
            elif acts == ['question']:
                tag = 'inquiry'
            elif not acts and prev_acts == ['question']:
                tag = 'inform'
            if tag:
                tagged_utterances.append((i, tag, utterance))

            prev_acts = acts

        return tagged_utterances

    def make_template(self, utterance, kb):
        """Abstract product and price.
        """
        title = kb.facts['item']['Title'].lower().split()
        new_tokens = []
        num_title = 0
        num_price = 0
        for token in utterance.tokens:
            if token in title:
                if len(new_tokens) > 0 and new_tokens[-1] == '{title}':
                    continue
                else:
                    num_title += 1
                    new_tokens.append('{title}')
            elif is_entity(token):
                num_price += 1
                new_tokens.append('{price}')
            else:
                new_tokens.append(token)

        if num_price > 1 or num_title > 1:
            return None
        template = self.detokenize(new_tokens)
        return template

    def extract_templates(self, transcripts_paths, max_examples=-1):
        templates = Trie()
        examples = read_examples(transcripts_paths, max_examples, Scenario)
        for example in examples:
            kbs = example.scenario.kbs
            events = example.events
            category = kbs[0].category
            roles = {agent_id: kbs[agent_id].role for agent_id in (0, 1)}

            utterances = [Utterance.from_text(event.data, self.price_tracker, kbs[event.agent]) for event in events]
            tagged_utterances = self.tag_utterances(utterances)

            for i, tag, utterance in tagged_utterances:
                template = self.make_template(utterance)
                role = roles[events[i].agent]
                if template:
                    key = (category, role, tag)
                    if key in templates:
                        templates[key].append(utterance)
                    else:
                        templates[key] = [utterance]
        return templates

############# TEST #############
if __name__ == '__main__':
    import argparse
    from core.price_tracker import PriceTracker
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcripts', nargs='*')
    parser.add_argument('--price-tracker-model')
    args = parser.parse_args()

    price_tracker = PriceTracker(args.price_tracker_model)
    template_extractor = TemplateExtractor(price_tracker)
    templates = template_extractor.extract_templates(args.transcripts, 10)
    for k, v in templates:
        print k
        for u in v:
            print u
