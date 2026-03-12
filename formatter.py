def format_recipe(example):

    ingredients = "\n".join(
        [f"{ing['ingredient']} - {ing['quantity']}"for ing in example["ingredients"]]
    )

    text = f"""
### Titolo
{example["title"]}

### Ingredienti
{ingredients}

### Preperazione
{example["preparation"]}
"""
    return {"text": text}
