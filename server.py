from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get('user_input')
    # Basic parsing for demonstration
    movie_title = user_input.split('like')[-1].strip() 
    recommendations = get_recommendations(movie_title)
    gpt_prompt = f"The user asked: '{user_input}'. Based on their request, I recommend: {','.join(recommendations)}."
    response = chat_with_gpt(gpt_prompt)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)