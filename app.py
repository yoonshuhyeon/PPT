
from flask import Flask, request, jsonify, render_template
import re
import os
import random
import json
import atexit

# --- 레벤슈타인 거리 계산 함수 ---
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1; deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# --- 지능의 구성 요소 ---
class LearningModule:
    def __init__(self, knowledge_file):
        self.knowledge_file = knowledge_file
        self.learning_pattern = r'^(.+?)(?:은|는) (.+?)$'
        self.privacy_keywords = ['이름', '주소', '번호', '나이', '직업', '학교', '회사']
        self.verb_endings = ['이다', '이야', '입니다', '야', '이']
        self.knowledge_db = self._load_from_file(self.knowledge_file, "지식")
    def _load_from_file(self, fp, mt):
        print(f"  > [{mt} 회로] 파일에서 {mt}을(를) 불러옵니다: {fp}")
        if os.path.exists(fp):
            try:
                with open(fp, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except json.JSONDecodeError: return {}
        return {}
    def _save_to_file(self, data, fp): json.dump(data, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    def process(self, text):
        subject_text = text.split('는')[0].split('은')[0]
        if any(keyword in subject_text for keyword in self.privacy_keywords) and ('내' in subject_text or '제' in subject_text):
            return "음... 그건 조금 사적인 주제 같네요. 다른 이야기를 해볼까요?", False
        match = re.match(self.learning_pattern, text)
        if not match: return None, False
        subject, raw_predicate = [s.strip() for s in match.groups()]
        predicate = raw_predicate
        for ending in self.verb_endings:
            if predicate.endswith(ending):
                predicate = predicate[:-len(ending)].strip(); break
        if predicate.endswith('.'): predicate = predicate[:-1]
        if subject in self.knowledge_db:
            if predicate not in self.knowledge_db[subject]:
                self.knowledge_db[subject].append(predicate); self._save_to_file(self.knowledge_db, self.knowledge_file)
                return f"(추가 지식 습득!) '{subject}'에 대한 새로운 사실을 배웠습니다.", True
            else: return f"그건 이미 알고 있어요. {subject}에 대한 사실 중 하나입니다.", False
        else:
            self.knowledge_db[subject] = [predicate]; self._save_to_file(self.knowledge_db, self.knowledge_file)
            return f"(지식 습득!) '{subject}'에 대한 사실을 배웠습니다.", True

class GenerativeModule:
    def __init__(self, memory_file):
        self.memory_file = memory_file
        self.markov_model = self._load_from_file(self.memory_file, "경험")
    def _load_from_file(self, fp, mt):
        print(f"  > [{mt} 회로] 파일에서 {mt}을(를) 불러옵니다: {fp}")
        if os.path.exists(fp):
            try:
                with open(fp, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except json.JSONDecodeError: return {}
        return {}
    def _save_to_file(self, data, fp): json.dump(data, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    def train(self, text):
        words = text.split()
        for i in range(len(words) - 1):
            self.markov_model.setdefault(words[i], []).append(words[i+1])

class SimpleChatBot:
    def __init__(self, kf="data/knowledge_v2.json", mf="data/memory_v2.json"):
        self.learning = LearningModule(kf)
        self.generative = GenerativeModule(mf)
        self.default_rules = [
            (r'^(안녕|하이|반가워|ㅎㅇ)[.!?]?$', ['반갑습니다.']),
            (r'(너의|네) 이름', ['PPT(Personal Pre-trained Transformer)입니다.']),
            (r'^(고마워|감사합니다|땡큐|감사)[.!?]?$', ['천만에요!']),
            (r'^(아하|아|오|그렇구나|알겠습니다|네)[.!?]?$', ['도움이 되셨다니 다행입니다.'])
        ]
    def think(self, user_input):
        self.generative.train(user_input)
        learning_response, learned = self.learning.process(user_input)
        if learned or (learning_response and not learned): return learning_response
        matched_subject = None
        for subject in self.learning.knowledge_db.keys():
            if user_input.startswith(subject):
                if matched_subject is None or len(subject) > len(matched_subject): matched_subject = subject
        if matched_subject:
            responses = self.learning.knowledge_db[matched_subject]
            return random.choice(responses)
        for pattern, responses in self.default_rules:
            if re.search(pattern, user_input): return random.choice(responses)
        scores = [(s, levenshtein_distance(user_input, s)) for s in self.learning.knowledge_db.keys()]
        best_matches = [s for s, dist in scores if dist <= len(s) / 3 and dist > 0]
        if best_matches:
            closest_match = min(best_matches, key=lambda s: levenshtein_distance(user_input, s))
            return f"혹시 '{closest_match}'에 대해 물어보셨나요?"
        return "그건 잘 모르겠어요. 'A는 B이다' 형식으로 알려주시겠어요?"
    def shutdown(self):
        print("\nAI의 기억을 파일에 저장합니다...")
        self.generative._save_to_file(self.generative.markov_model, self.generative.memory_file)
        print("저장 완료.")

# --- Flask 웹 애플리케이션 설정 ---
app = Flask(__name__, template_folder='.')
print("AI의 의식을 로딩합니다...")
chatbot = SimpleChatBot()
print("AI 로딩 완료.")
atexit.register(chatbot.shutdown)

@app.route('/')
def home(): return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    bot_response = chatbot.think(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
