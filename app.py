
from flask import Flask, request, jsonify, render_template
import re
import os
import random
import json
import math
import atexit

# --- 지능의 구성 요소 (이전과 동일) ---
class LearningModule:
    def __init__(self, knowledge_file):
        self.knowledge_file = knowledge_file
        self.learning_pattern = r'^(.+?)(?:은|는) (.+?)(?:이다|이야|입니다)?\.?$'
        self.privacy_keywords = ['이름', '주소', '번호', '나이', '직업', '학교', '회사']
        self.learned_rules = self._load_from_file(self.knowledge_file, "지식")
    def _load_from_file(self, fp, mt): return json.load(open(fp, 'r', encoding='utf-8')) if os.path.exists(fp) else []
    def _save_to_file(self, data, fp): json.dump(data, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    def process(self, text):
        subject_text = text.split('는')[0].split('은')[0]
        if any(keyword in subject_text for keyword in self.privacy_keywords):
            return "(개인정보로 판단되어 학습하지 않습니다.)", self.learned_rules
        match = re.match(self.learning_pattern, text)
        if not match: return None, None
        subject, predicate = [s.strip() for s in match.groups()]
        new_pattern = f'^{re.escape(subject)}(?:란|이란|는|은)?\??$'
        if any(p == new_pattern for p, _ in self.learned_rules): return None, None
        self.learned_rules.append((new_pattern, [predicate]))
        self._save_to_file(self.learned_rules, self.knowledge_file)
        return f"(공유 지식 습득) '{subject}'에 대한 사실을 모두와 공유합니다.", self.learned_rules

class AttentionModule:
    def __init__(self): self.d_model,self.d_k,self.W_q,self.W_k,self.W_v,self.embedding_cache=4,3,[[1,1,0,1],[0,1,1,0],[1,0,1,1]],[[1,0,1,1],[1,1,0,0],[0,1,1,1]],[[0,2,0,1],[1,0,3,0],[0,1,1,2]],{}
    def _get_embedding(self, word): 
        if word not in self.embedding_cache: self.embedding_cache[word] = [random.uniform(-1, 1) for _ in range(self.d_model)]
        return self.embedding_cache[word]
    def process(self, text):
        words = text.split()
        if not (1 < len(words) < 8): return "(분석 실패)", None
        embeddings = {w: self._get_embedding(w) for w in words}; queries = {w: matrix_vector_multiply(self.W_q, e) for w, e in embeddings.items()}; keys = {w: matrix_vector_multiply(self.W_k, e) for w, e in embeddings.items()}
        focus_word = words[0]; scores = [dot_product(queries[focus_word], keys[w]) for w in words]; weights = softmax([s / math.sqrt(self.d_k) for s in scores])
        analysis_str = f"(분석) '{focus_word}'는 다른 단어에 각각 {[f'{w:.1%}' for w in weights]} 만큼 주목합니다."
        target_word = None
        if len(weights) > 1: temp_weights = weights[:]; temp_weights[0] = -1.0; max_w = max(temp_weights)
        if max_w > 0: target_word = words[temp_weights.index(max_w)]
        return analysis_str, target_word

class GenerativeModule:
    def __init__(self, memory_file): self.memory_file = memory_file; self.markov_model = self._load_from_file(self.memory_file, "경험")
    def _load_from_file(self, fp, mt): return json.load(open(fp, 'r', encoding='utf-8')) if os.path.exists(fp) else {}
    def _save_to_file(self, data, fp): json.dump(data, open(fp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    def train(self, text): words = text.split(); [self.markov_model.setdefault(words[i], []).append(words[i+1]) for i in range(len(words) - 1)]
    def process(self, text, length=8):
        if not self.markov_model: return "(아직 대화 기록이 부족하여 추론할 수 없습니다.)"
        words = text.split(); start_word = max(set(words), key=lambda w: len(self.markov_model.get(w, [])), default=None)
        if not start_word or start_word not in self.markov_model:
            all_words = [word for sublist in self.markov_model.values() for word in sublist]
            if not all_words: return "(...)"
            start_word = max(set(all_words), key=all_words.count)
        result = [start_word]
        for _ in range(length - 1):
            if result[-1] in self.markov_model: result.append(max(set(self.markov_model[result[-1]]), key=self.markov_model[result[-1]].count))
            else: break
        return f"(기억 종합) ... {' '.join(result)} ..."

class SharedBrainChatBot:
    def __init__(self, kf="data/final_bot_knowledge.json", mf="data/final_bot_memory.json"): self.learning = LearningModule(kf); self.attention = AttentionModule(); self.generative = GenerativeModule(mf); self.default_rules = [(r'안녕|하이',['반갑습니다.']),(r'너의 이름|네 이름',['저는 모두의 지식으로 성장하는 AI입니다.'])]
    def think(self, user_input): self.generative.train(user_input); learning_response, _ = self.learning.process(user_input); 
         if learning_response: return learning_response
        for pattern, responses in self.learning.learned_rules + self.default_rules:
            if re.search(pattern, user_input): return random.choice(responses)
        analysis_result, target_word = self.attention.process(user_input)
        if target_word: return f"{analysis_result}\n(사고) '{target_word}'에 주목하여 생각을 이어갑니다.\n{self.generative.process(target_word)}"
        else:    return f"{analysis_result}\n{self.generative.process(user_input)}"
    def shutdown(self): self.generative._save_to_file(self.generative.markov_model, self.generative.memory_file)

# --- Flask 웹 애플리케이션 설정 ---

app = Flask(__name__, template_folder='.') # Gunicorn이 찾는 app 객체

print("공유 지능 AI의 의식을 로딩합니다...")
chatbot = SharedBrainChatBot()
print("AI 로딩 완료.")

# 서버 종료 시, 챗봇의 기억을 저장하는 함수 등록
atexit.register(chatbot.shutdown)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    bot_response = chatbot.think(user_message)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    # 로컬에서 테스트할 때 사용하는 실행 방식입니다.
    app.run(host='0.0.0.0', port=8000)
