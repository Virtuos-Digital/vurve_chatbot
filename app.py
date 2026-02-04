from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.utils import simple_preprocess

app = Flask(__name__)

# Load model and artifacts on startup
print("Loading model and artifacts...")
loaded_model = load_model('lstm_intent_classifier_9kdataset.keras')
with open('tokenizer_9kdataset.pkl', 'rb') as f:
    loaded_tokenizer = pickle.load(f)
with open('model_config_9kdataset.pkl', 'rb') as f:
    loaded_config = pickle.load(f)
print("✓ Model ready!")

def preprocess_text(text):
    tokens = simple_preprocess(text, deacc=True)
    return tokens

def predict_intent(sentence):
    tokens = preprocess_text(sentence)
    preprocessed_text = ' '.join(tokens)
    
    sequence = loaded_tokenizer.texts_to_sequences([preprocessed_text])
    padded = pad_sequences(
        sequence, 
        maxlen=loaded_config['max_length'], 
        padding='post', 
        truncating='post'
    )
    
    prediction = loaded_model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    reverse_label_mapping = {0: -2, 1: -1, 2: 1, 3: 2}
    original_label = reverse_label_mapping[predicted_class]
    
    label_descriptions = {
        -2: "Escalate to human agent",
        -1: "Track order/order status (login required)",
        1: "General conversation/company policy",
        2: "Product recommendation"
    }
    
    return {
        'sentence': sentence,
        'label': int(original_label),
        'description': label_descriptions[original_label],
        'confidence': float(confidence),
        'probabilities': {
            'Escalate': float(prediction[0][0]),
            'Track order': float(prediction[0][1]),
            'Conversation': float(prediction[0][2]),
            'Recommendation': float(prediction[0][3])
        }
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'sentence' not in data:
            return jsonify({'error': 'Missing "sentence" field in request'}), 400
        
        sentence = data['sentence']
        if not isinstance(sentence, str) or len(sentence.strip()) == 0:
            return jsonify({'error': 'Sentence must be a non-empty string'}), 400
        
        result = predict_intent(sentence)
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # For local development only. For production, use a WSGI server like Gunicorn.
    app.run(debug=True, host='0.0.0.0', port=5000)
