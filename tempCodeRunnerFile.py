from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

df_recipes = pd.read_csv("D:/minipro5/archive/recipes.csv")

tfidf = TfidfVectorizer(stop_words="english")
df_recipes['ingredients'] = df_recipes['ingredients'].fillna("")
tfidf_matrix = tfidf.fit_transform(df_recipes['ingredients'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

recommended_recipes_set = set()

def get_recommendations(user_ingredients):
    user_tfidf = tfidf.transform([" ".join(user_ingredients)])
    user_cosine_sim = linear_kernel(user_tfidf, tfidf_matrix).flatten()
    recipe_indices = user_cosine_sim.argsort()[::-1]

    data = []

    for idx in recipe_indices:
        recipe_name = df_recipes['recipe_name'].iloc[idx]
        recipe_img = df_recipes['img_src'].iloc[idx]
        recipe_rating = df_recipes['rating'].iloc[idx]
        recipe_url = df_recipes['url'].iloc[idx]

        if recipe_name not in recommended_recipes_set:
            recommended_recipes_set.add(recipe_name)
            data.append([recipe_name, recipe_img, recipe_rating, recipe_url])

        if len(recommended_recipes_set) >= 10:
            break
    recommended_recipes_set.clear()
    return data

@app.route('/')
def index():
    return render_template('index.html', 
                           recipe_name=list(df_recipes['recipe_name'].values),
                           img=list(df_recipes['img_src'].values),
                           time=list(df_recipes['total_time'].values),
                           rating=list(df_recipes['rating'].values))

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_ui():
    try:
        if request.method == 'POST':
            user_input = request.form.get('user_ingredients')
            if user_input:
                user_ingredients = [ingredient.strip() for ingredient in user_input.split(",")]

                recommended_data = get_recommendations(user_ingredients)

                return render_template('recommend.html', recommended_data=recommended_data)

        return render_template('recommend.html')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
