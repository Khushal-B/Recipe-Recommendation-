# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel

# app = Flask(__name__)

# df_recipes = pd.read_csv("./archive/recipes.csv")

# tfidf = TfidfVectorizer(stop_words="english")
# df_recipes['ingredients'] = df_recipes['ingredients'].fillna("")
# tfidf_matrix = tfidf.fit_transform(df_recipes['ingredients'])
# cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# recommended_recipes_set = set()

# def get_recommendations(user_ingredients):
#     user_tfidf = tfidf.transform([" ".join(user_ingredients)])
#     user_cosine_sim = linear_kernel(user_tfidf, tfidf_matrix).flatten()
#     recipe_indices = user_cosine_sim.argsort()[::-1]

#     data = []

#     for idx in recipe_indices:
#         recipe_name = df_recipes['recipe_name'].iloc[idx]
#         recipe_img = df_recipes['img_src'].iloc[idx]
#         recipe_rating = df_recipes['rating'].iloc[idx]
#         recipe_url = df_recipes['url'].iloc[idx]

#         if recipe_name not in recommended_recipes_set:
#             recommended_recipes_set.add(recipe_name)
#             data.append([recipe_name, recipe_img, recipe_rating, recipe_url])

#         if len(recommended_recipes_set) >= 10:
#             break
#     recommended_recipes_set.clear()
#     return data

# @app.route('/')
# def index():
#     return render_template('index.html', 
#                            recipe_name=list(df_recipes['recipe_name'].values),
#                            img=list(df_recipes['img_src'].values),
#                            time=list(df_recipes['total_time'].values),
#                            rating=list(df_recipes['rating'].values))

# @app.route('/recommend', methods=['GET', 'POST'])
# def recommend_ui():
#     try:
#         if request.method == 'POST':
#             user_input = request.form.get('user_ingredients')
#             if user_input:
#                 user_ingredients = [ingredient.strip() for ingredient in user_input.split(",")]

#                 recommended_data = get_recommendations(user_ingredients)

#                 return render_template('recommend.html', recommended_data=recommended_data)

#         return render_template('recommend.html')

#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

df_recipes = pd.read_csv("./archive/recipes.csv")

# Set up TF-IDF vectorizer and cosine similarity matrix
tfidf = TfidfVectorizer(stop_words="english")
df_recipes['ingredients'] = df_recipes['ingredients'].fillna("")
tfidf_matrix = tfidf.fit_transform(df_recipes['ingredients'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

recommended_recipes_set = set()

def get_recommendations(user_ingredients, condition, allergies):
    user_tfidf = tfidf.transform([" ".join(user_ingredients)])
    user_cosine_sim = linear_kernel(user_tfidf, tfidf_matrix).flatten()
    recipe_indices = user_cosine_sim.argsort()[::-1]

    data = []

    for idx in recipe_indices:
        recipe = df_recipes.iloc[idx]
        
        # Filter based on condition
        if condition == "diabetes" and "Total Sugars" in recipe['nutrition']:
            sugar_content = int(recipe['nutrition'].split("Total Sugars")[1].split("g")[0].strip())
            if sugar_content > 10:
                continue
        elif condition == "high_cholesterol" and "Cholesterol" in recipe['nutrition']:
            cholesterol_content = int(recipe['nutrition'].split("Cholesterol")[1].split("mg")[0].strip())
            if cholesterol_content > 50:
                continue

        # Filter based on allergies
        if any(allergen.lower() in recipe['ingredients'].lower() for allergen in allergies):
            continue

        if recipe['recipe_name'] not in recommended_recipes_set:
            recommended_recipes_set.add(recipe['recipe_name'])
            data.append({
                "name": recipe['recipe_name'],
                "img": recipe['img_src'],
                "rating": recipe['rating'],
                "url": recipe['url']
            })

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
            condition = request.form.get('condition')
            allergies = request.form.getlist('allergies')
            
            if user_input:
                user_ingredients = [ingredient.strip() for ingredient in user_input.split(",")]
                recommended_data = get_recommendations(user_ingredients, condition, allergies)

                return render_template('recommend.html', recommended_data=recommended_data, condition=condition)

        return render_template('recommend.html')

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(debug=True)
