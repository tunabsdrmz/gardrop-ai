# Evaluation

A small, honest benchmark for `/tag`: how often does it get the category, pattern and formality right?

1. Take 40-60 photos of your own clothes (about 10 per category, mixed backgrounds).
   Don't use photos from shops or the internet; you don't have the rights to publish them.
2. Put them in `eval/images/`.
3. Add one line per photo to `eval/labels.csv`:
   ```csv
   filename,category,pattern,formality
   jeans-01.jpg,bottom,solid,casual
   blazer-02.jpg,outerwear,solid,smart-casual
   ```
   Allowed values are the ones in `app/schemas.py`. Leave a column empty if unsure.
4. Run `uv run python -m eval.evaluate`.

The script prints an accuracy table and a category confusion matrix, and saves them to `eval/results.md`. Paste the table into the main README.
